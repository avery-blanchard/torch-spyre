/*
 * Copyright 2025 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "spyre_mem.h"

#include <ATen/EmptyTensor.h>
#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/set_cpu_dispatch.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <pybind11/pybind11.h>
#include <torch/library.h>

#include <algorithm>
#include <climits>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "logging.h"
#include "module.h"
#include "spyre_allocator.h"
#include "spyre_composite_address.h"
#include "spyre_storage_impl.h"
#include "spyre_stream.h"
#include "spyre_tensor_impl.h"
#include "types_mapping.h"

namespace py = pybind11;

namespace spyre {

/*
 * Fills out size and strides for each dimension of the tensor.
 *
 * dcsi_sizes[j] is derived from stl.valid_elements:
 *   - singleton group {j}: dcsi_sizes[j] = min(valid_count, device_size[j])
 *   - grouped pair {outer, inner}: dcsi_sizes[inner] = min(V, inner_size),
 *     dcsi_sizes[outer] = V / inner_size
 * A remainder DCSI is emitted for each grouped pair where
 * valid_count % device_size[inner] != 0 (partial stick row).
 *
 * @param cpu_offset: storage offset of the CPU tensor
 * @param device_offset: storage offset of the dev tensor
 * @param stl: SpyreTensorLayout of dev tensor
 * @param host2device: direction of data conversion
 * @return description of data conversion
 */
auto get_device_stride_infos(int64_t cpu_offset, int64_t device_offset,
                             const SpyreTensorLayout& stl, bool host2device)
    -> std::vector<DataConversionStrideInfo> {
  const int device_rank = static_cast<int>(stl.stride_map.size());

  // host_strides[j] = stride_map[j] (the host-memory stride for device dim j).
  // device_strides[j] = contiguous product strides over device_size.
  std::vector<int64_t> host_strides(device_rank, 1);
  std::vector<int64_t> device_strides(device_rank, 1);

  int64_t prev_size = 1;
  for (int i = device_rank - 1; i >= 0; i--) {
    device_strides[i] = prev_size;
    prev_size *= stl.device_size[i];
    if (stl.stride_map[i] > 0) {
      host_strides[i] = stl.stride_map[i];
    }
  }

  // dcsi_sizes[j] = number of valid elements in device dim j.
  // For broadcast dims (stride_map==0) use the full device extent.
  // For size-1 / sparse dims (stride_map==-1) leave as 1.
  // For singleton groups: dcsi_sizes[j] = min(valid_count, device_size[j]).
  // For grouped (stick) pairs {outer, inner} with valid_count V:
  //   dcsi_sizes[inner] = min(inner_size, V)  (full inner row)
  //   dcsi_sizes[outer] = V / inner_size       (number of full rows)
  std::vector<int64_t> dcsi_sizes(device_rank, 1);
  std::vector<bool> set_by_group(device_rank, false);

  // Remainder info: at most one partial-stick-row DCSI can arise (enforced by
  // TORCH_CHECK below).  Stored as an optional to avoid heap allocation.
  struct RemainderInfo {
    std::vector<int64_t> size_overrides;  // device_rank entries, reversed later
    int64_t host_off;
    int64_t dev_off;
  };
  std::optional<RemainderInfo> remainder_opt;

  // First pass: grouped {outer, inner} stick-pair keys.
  // Computes dcsi_sizes for both dims and captures any partial-row remainder.
  for (const auto& [dims, valid_count] : stl.valid_elements) {
    if (dims.size() < 2) continue;
    TORCH_CHECK(dims.size() == 2, "valid_elements key with ", dims.size(),
                " dims is not supported"
                "; only singleton {j} and stick-pair {outer, inner} keys are"
                " valid");
    const int inner = static_cast<int>(dims.back());
    const int outer = static_cast<int>(dims[dims.size() - 2]);
    const int64_t inner_size = stl.device_size[inner];
    TORCH_CHECK(inner_size > 0, "Invalid device size ", inner_size,
                " for device dimension ", inner);
    const int64_t tiled = valid_count / inner_size;
    dcsi_sizes[inner] = std::min(valid_count, inner_size);
    dcsi_sizes[outer] = tiled;
    set_by_group[inner] = true;
    set_by_group[outer] = true;

    const int64_t leftover = valid_count % inner_size;
    if (leftover != 0) {
      std::vector<int64_t> overrides(device_rank, 0);
      overrides[inner] = leftover;
      overrides[outer] = 1;
      remainder_opt =
          RemainderInfo{std::move(overrides), tiled * host_strides[outer],
                        tiled * device_strides[outer]};
    }
  }

  // Second pass: singleton groups and broadcast/sparse dims.
  const std::map<int64_t, int64_t> ve = stl.valid_elements_per_dim();
  for (int i = 0; i < device_rank; i++) {
    TORCH_CHECK(stl.device_size[i] >= 0, "Invalid device size ",
                stl.device_size[i], " for device dimension ", i);
    if (set_by_group[i]) continue;
    if (stl.stride_map[i] == 0) {
      dcsi_sizes[i] = stl.device_size[i];
    } else if (stl.stride_map[i] > 0) {
      const auto it = ve.find(static_cast<int64_t>(i));
      dcsi_sizes[i] = it != ve.end() ? std::min(it->second, stl.device_size[i])
                                     : stl.device_size[i];
    }
  }

  // Build the template DCSI (strides, offsets) used for both the full-row DCSI
  // and as the base from which any remainder DCSI is derived.
  DataConversionStrideInfo dcsi_template;
  dcsi_template.size_ = dcsi_sizes;
  dcsi_template.stride_src_ = host2device ? host_strides : device_strides;
  dcsi_template.stride_dst_ = host2device ? device_strides : host_strides;
  dcsi_template.offset_src_ = host2device ? cpu_offset : device_offset;
  dcsi_template.offset_dst_ = host2device ? device_offset : cpu_offset;
  std::reverse(dcsi_template.size_.begin(), dcsi_template.size_.end());
  std::reverse(dcsi_template.stride_src_.begin(),
               dcsi_template.stride_src_.end());
  std::reverse(dcsi_template.stride_dst_.begin(),
               dcsi_template.stride_dst_.end());

  // Emit the full-row DCSI only when no dcsi_sizes entry is zero.  When
  // valid_count < inner_size all valid elements fit in a single partial stick
  // row; in that case the only DCSI needed is the remainder one below.
  const bool has_full_rows = std::none_of(dcsi_sizes.begin(), dcsi_sizes.end(),
                                          [](int64_t s) { return s == 0; });

  std::vector<DataConversionStrideInfo> stride_infos;
  if (has_full_rows) stride_infos.push_back(dcsi_template);

  // Emit at most one remainder DCSI for the partial stick row (if any).
  if (remainder_opt) {
    auto& rem = *remainder_opt;
    std::reverse(rem.size_overrides.begin(), rem.size_overrides.end());
    const int64_t offset_src = host2device ? rem.host_off : rem.dev_off;
    const int64_t offset_dst = host2device ? rem.dev_off : rem.host_off;
    DataConversionStrideInfo info =
        stride_infos.empty() ? dcsi_template : stride_infos[0];
    for (int k = 0; k < device_rank; k++) {
      if (rem.size_overrides[k] != 0) info.size_[k] = rem.size_overrides[k];
    }
    info.offset_src_ += offset_src;
    info.offset_dst_ += offset_dst;
    stride_infos.push_back(info);
  }

  return stride_infos;
}

/*
 * Generate description of data conversion for a tensor.
 *
 * @param cpu_tensor: CPU-side tensor (source for H2D, destination for D2H)
 * @param dev_tensor: device-side tensor (destination for H2D, source for D2H)
 * @return data conversion information
 */
auto generate_dci(const at::Tensor* cpu_tensor, const at::Tensor* dev_tensor,
                  SpyreTensorLayout stl, bool host2device)
    -> DataConversionInfo {
  // Support dtype conversion: populate DCI with both source and destination
  // dtype formats
  auto cpu_str_type = torchScalarToString[cpu_tensor->scalar_type()];
  auto dev_str_type = torchScalarToString[dev_tensor->scalar_type()];
  const auto [cpu_format_host, cpu_format_dev] =
      stringToDTDataFormatPair(cpu_str_type);
  TORCH_CHECK(cpu_format_host != DataFormats::INVALID &&
                  cpu_format_dev != DataFormats::INVALID,
              "Unsupported CPU tensor dtype for DMA transfer: ", cpu_str_type);

  // stl.device_dtype is the authoritative on-device element format.  For bool
  // tensors it may differ from the type-map default (SEN169_FP16): an fp32
  // comparison result is physically stored as IEEE_FP32 (32 elems/stick).
  //
  // IEEE_FP32 bool tensors are on-device intermediates produced by the
  // compiler; they are always read D2H and never written H2D.  Guard against
  // that assumption being violated so a future regression fails loudly rather
  // than silently transferring data with the wrong element size.
  TORCH_CHECK(
      !(host2device && dev_tensor->scalar_type() == c10::ScalarType::Bool &&
        stl.device_dtype == DataFormats::IEEE_FP32),
      "Unexpected H2D transfer to a bool tensor with IEEE_FP32 device format; "
      "fp32-format bool tensors are on-device intermediates and should never "
      "be H2D transfer destinations");
  DataConversionInfo dci{};
  dci.dci_dsName_ = "DCI-Tensor-0";
  dci.isHostToSen_ = host2device;
  dci.dataformat_src_ = host2device ? cpu_format_host : stl.device_dtype;
  dci.dataformat_dst_ = host2device ? stl.device_dtype : cpu_format_host;
  TORCH_CHECK(
      isDCIConversionSupported(dci.dataformat_src_, dci.dataformat_dst_),
      "Unsupported DCI data format conversion: src=",
      static_cast<int>(dci.dataformat_src_),
      " dst=", static_cast<int>(dci.dataformat_dst_),
      " (cpu_type=", cpu_str_type, ", dev_type=", dev_str_type, ")");

  std::vector<int64_t> cpu_sizes = cpu_tensor->sizes().vec();
  std::vector<int64_t> cpu_strides = cpu_tensor->strides().vec();
  std::vector<int64_t> dev_sizes = dev_tensor->sizes().vec();
  std::vector<int64_t> dev_strides = dev_tensor->strides().vec();
  std::vector<int64_t> device_sizes = stl.device_size;

  // While the source strides may differ from the destination strides when the
  // source is non-dense or overlapping, the source sizes should always match
  // the destination sizes.
  //
  // This is assumed to be true for the following logic, so this check should
  // not be removed unless the following logic is updated accordingly.
  TORCH_CHECK(cpu_sizes == dev_sizes,
              "Invalid device sizes for host sizes. Expected: ", cpu_sizes,
              ", got: ", dev_sizes);

  const int64_t cpu_offset = cpu_tensor->storage_offset();
  int64_t dev_offset = dev_tensor->storage_offset();
  int64_t device_offset = 0;

  const int device_rank = static_cast<int>(stl.stride_map.size());

  // Compute device_offset (element offset into the device_size coordinate
  // space) from dev_offset (the tensor's element storage offset).
  std::vector<int64_t> device_strides(device_rank, 1);
  int64_t device_elements = 1;
  for (int i = device_rank - 1; i >= 0; i--) {
    device_strides[i] = device_elements;
    device_elements *= stl.device_size[i];
  }

  std::map<int64_t, int, std::greater<int64_t>> stride_map_to_j;
  for (int i = 0; i < device_rank; i++) {
    if (stl.stride_map[i] <= 0) continue;
    if (stl.device_size[i] <= 1) continue;
    if (stride_map_to_j.contains(stl.stride_map[i])) {
      TORCH_CHECK(i == (device_rank - 1),
                  "Invalid device sizes and stride map. Expected only 1 non-1 ",
                  "device_size value for stride_map value ", stl.stride_map[i],
                  "got: ", stl.device_size[i], " and ",
                  stl.device_size[stride_map_to_j[stl.stride_map[i]]]);
      continue;
    }
    stride_map_to_j.insert({stl.stride_map[i], i});
  }

  for (const auto& [stride, index] : stride_map_to_j) {
    if (stride > dev_offset) continue;
    const int64_t slice = dev_offset / stride;
    device_offset += slice * device_strides[index];
    dev_offset -= slice * stride;
    if (dev_offset <= 0) break;
  }

  TORCH_CHECK(dev_offset == 0, "Invalid device tensor storage offset");
  TORCH_CHECK(device_offset < device_elements, "Invalid device storage offset");

  // Cap valid_elements by the actual tensor sizes so that a sliced tensor
  // is described correctly by its DCSIs.  When the allocation is larger than
  // the logical tensor (e.g., a view into a padded allocation), valid_elements
  // reflects the full allocation and must be trimmed to what is actually
  // being transferred.
  //
  // Each valid_elements group corresponds to one or more host dimensions that
  // share a stick boundary.  The host dimension count for a group is found via
  // its finest host-memory stride: the smallest positive stride_map value in
  // that group identifies the host dimension whose logical size should cap the
  // group's valid_count.  (For a stick-paired group {outer, inner}, the inner
  // device dim's stride_map value is the host stick-dim stride — the finest
  // granularity — and dev_strides[stick_dim] gives the logical row count.)
  //
  // Only positive dev_strides are used as keys so that H2D-expanded dims
  // (cpu_stride==0) are excluded — their device dim handling is in the H2D
  // block below.
  {
    // Build alloc_strides: the set of positive stride_map values for the
    // allocation. Used to determine whether a dev_strides value corresponds to
    // an allocation dimension.
    std::set<int64_t> alloc_strides;
    for (int i = 0; i < device_rank; i++) {
      const int64_t sm = stl.stride_map[i];
      if (sm > 0) alloc_strides.insert(sm);
    }

    const int host_rank = static_cast<int>(dev_sizes.size());
    // In a single pass over dev_strides, build:
    //   dev_stride_set: all positive dev strides (for absent-stride detection).
    //   stride_to_host_size: alloc stride → logical size, but only for dims
    //     whose every coarser dev stride is also an alloc stride.  This guards
    //     against views that introduce extra outer dims above an alloc stride
    //     (e.g. b.view(64,8,512) adds stride-4096 above stride-512) and would
    //     otherwise incorrectly cap the valid_count for the stride-512 group.
    std::set<int64_t> dev_stride_set;
    std::map<int64_t, int64_t> stride_to_host_size;
    for (int i = 0; i < host_rank; i++) {
      const int64_t s = dev_strides[i];
      if (s <= 0) continue;
      dev_stride_set.insert(s);
      if (alloc_strides.find(s) == alloc_strides.end()) continue;
      bool coarser_all_in_alloc = true;
      for (int j = 0; j < host_rank; j++) {
        if (dev_strides[j] > s &&
            alloc_strides.find(dev_strides[j]) == alloc_strides.end()) {
          coarser_all_in_alloc = false;
          break;
        }
      }
      if (!coarser_all_in_alloc) continue;
      // Use max to handle duplicate strides (e.g. unsqueeze(-1) adds a size-1
      // dim with the same stride as the real inner dim); the larger size wins.
      auto [sit, sinserted] = stride_to_host_size.emplace(s, dev_sizes[i]);
      if (!sinserted) sit->second = std::max(sit->second, dev_sizes[i]);
    }

    for (auto& [dims, valid_count] : stl.valid_elements) {
      int64_t min_pos_stride = INT64_MAX;
      for (int64_t d : dims) {
        const int64_t sm = stl.stride_map[static_cast<int>(d)];
        if (sm > 0) min_pos_stride = std::min(min_pos_stride, sm);
      }
      if (min_pos_stride == INT64_MAX) continue;
      // If this allocation stride is entirely absent from dev_strides, the
      // logical view has been rank-reduced (indexed) along that dimension.
      // The tensor is fixed at one position in that dimension, so only 1
      // element (or 1 row for a grouped key) is valid.
      if (dev_stride_set.find(min_pos_stride) == dev_stride_set.end()) {
        valid_count = 1;
        continue;
      }
      const auto it = stride_to_host_size.find(min_pos_stride);
      if (it == stride_to_host_size.end()) continue;
      valid_count = std::min(valid_count, it->second);
    }
  }

  if (host2device) {
    // If the dev_strides do not match the cpu_strides then the cpu_tensor is
    // sliced and/or expanded.
    if (cpu_strides != dev_strides) {
      // Update the local copy of stride_map to reflect cpu_strides.
      // The SpyreTensorImpl for the dev_tensor is not modified.
      //
      // NOTE: stl.valid_elements is intentionally NOT updated here.
      // get_device_stride_infos only consults valid_elements for the sizes of
      // dims with positive stride_map entries.  Expanded dims have their
      // stride_map entries zeroed below, so get_device_stride_infos will use
      // the full device_size for those dims regardless of valid_elements — the
      // valid_elements group membership for the original allocation is harmless
      // because it is only read for dims that remain positive in stride_map.
      const int host_rank = cpu_sizes.size();

      std::vector<int64_t> dst_stride_map = stl.stride_map;
      std::vector<int64_t> dst_strides = dev_strides;

      // For expanded dims (cpu_stride==0), zero the stride_map entry for every
      // device dim whose stride_map value matches the expanded host dim's
      // stride.
      std::map<int64_t, int64_t> expands;
      for (int i = 0; i < host_rank; i++) {
        const int64_t cpu_stride = cpu_strides[i];
        const int64_t dev_stride = dev_strides[i];
        if (cpu_stride == 0) {
          const int64_t expand = cpu_sizes[i];
          expands.insert({dev_stride, expand});
          for (int j = 0; j < device_rank; j++) {
            if (stl.stride_map[j] == dev_stride) {
              dst_stride_map[j] = 0;
            }
          }
        }
      }

      for (int i = 0; i < device_rank; i++) {
        if (dst_stride_map[i] <= 0) continue;
        for (const auto& [stride, expand] : expands) {
          if (stl.stride_map[i] <= stride) continue;
          dst_stride_map[i] /= expand;
        }
      }

      for (int i = 0; i < host_rank; i++) {
        const int64_t dev_stride = dev_strides[i];
        for (const auto& [stride, expand] : expands) {
          if (dev_stride <= stride) continue;
          dst_strides[i] /= expand;
        }
      }

      stl.stride_map = dst_stride_map;

      // Adjust the dst_stride_map for all sliced dimension by multiplying all
      // other stride values greater than or equal to the sliced stride value
      // by the amount sliced.
      std::vector<int64_t> cpu_order(host_rank);
      std::iota(cpu_order.begin(), cpu_order.end(), 0);
      std::sort(cpu_order.begin(), cpu_order.end(),
                [&cpu_strides, &cpu_sizes](int64_t i1, int64_t i2) {
                  if (cpu_strides[i1] == cpu_strides[i2]) {
                    if (cpu_sizes[i1] == 1 && cpu_sizes[i2] == 1) {
                      return i1 < i2;
                    }
                    return cpu_sizes[i1] == 1;
                  }
                  return cpu_strides[i1] < cpu_strides[i2];
                });

      std::multimap<int64_t, int64_t> slices;
      int64_t current_slices = 1;
      for (const auto& i : cpu_order) {
        const int64_t cpu_stride = cpu_strides[i];
        const int64_t dev_stride = dst_strides[i];
        TORCH_CHECK(
            dev_stride > 0,
            "Invalid destination stride. Expected > 0, got: ", dev_stride);
        const int64_t slice = cpu_stride / dev_stride / current_slices;
        if (slice > 1) {
          slices.insert({dev_stride, slice});
          current_slices *= slice;
        }
      }

      for (int i = 0; i < device_rank; i++) {
        if (dst_stride_map[i] <= 0) continue;
        for (const auto& [stride, slice] : slices) {
          if (stl.stride_map[i] < stride) continue;
          dst_stride_map[i] *= slice;
        }
      }

      stl.stride_map = dst_stride_map;
    }
  }
  // FP8 multi-dim stick layout uses specialized DCI generation
  if (stl.element_arrangement == ElementArrangement::QFP8WT) {
    // Reverse PyTorch ordering before reading N/K from cpu_sizes.
    std::reverse(cpu_sizes.begin(), cpu_sizes.end());
    std::reverse(device_sizes.begin(), device_sizes.end());

    const int64_t eps = stl.elems_per_stick();
    const int64_t si = 2;
    const int64_t so = eps / si;
    // cpu_sizes is reversed from PyTorch ordering [K, N] → [N, K], so
    // cpu_sizes[0] = N (columns) and cpu_sizes[1] = K (rows).
    const int64_t N = cpu_sizes[0];
    const int64_t K = cpu_sizes[1];

    // Expanded device shape: [si, so, K/si, N/so]
    TORCH_CHECK(K % si == 0, "QFP8WT K=", K, " must be divisible by si=", si);
    TORCH_CHECK(N % so == 0, "QFP8WT N=", N, " must be divisible by so=", so);
    const int64_t dim2 = K / si;
    const int64_t dim3 = N / so;

    // Host strides for the expanded layout
    const int64_t dst2 = si * so;  // = eps = 128 (one full [2,64] stick)
    const int64_t dst3 =
        dim2 * eps;  // = (K/si) * si * so = K * so (one N-strip of sticks)

    // Host (row-major) strides in the expanded 4D index space [si, so, dim2,
    // dim3]:
    //   element [k,n]: k = d2*si + a,  n = d3*so + b
    //   host byte = k*N + n = (d2*si + a)*N + (d3*so + b)
    //             = a*N + b*1 + d2*(si*N) + d3*so
    //   → stride_src = [N, 1, si*N, so]
    const std::vector<int64_t> host_strides = {N, 1, si * N, so};

    // Device (QFP8WT [2,64] stick) strides in the expanded 4D index space:
    //   physical byte = a*1 + b*si + d2*dst2 + d3*dst3
    //   (within a stick: si dimension innermost at stride=1, so dimension at
    //   stride=si) → stride = [1, si, dst2, dst3]
    const std::vector<int64_t> device_strides = {1, si, dst2, dst3};

    DataConversionStrideInfo dcsi;
    dcsi.size_ = {si, so, dim2, dim3};
    dcsi.stride_src_ = host2device ? host_strides : device_strides;
    dcsi.stride_dst_ = host2device ? device_strides : host_strides;
    dcsi.offset_src_ = host2device ? cpu_offset : 0;
    dcsi.offset_dst_ = host2device ? 0 : cpu_offset;
    dci.dcsi_ = {dcsi};
    // For H2D, output (device) shape is the expanded 4D form.
    // For D2H, output (host) shape is the original 2D cpu_sizes.
    const std::vector<int64_t> expanded_dev_shape = {si, so, dim2, dim3};
    dci.output_shape_ = host2device ? expanded_dev_shape : cpu_sizes;
    dci.input_shape_ = host2device ? cpu_sizes : device_sizes;
  } else {
    dci.dcsi_ =
        get_device_stride_infos(cpu_offset, device_offset, stl, host2device);

    // Reverse PyTorch ordering
    std::reverse(cpu_sizes.begin(), cpu_sizes.end());
    std::reverse(device_sizes.begin(), device_sizes.end());
    dci.input_shape_ = host2device ? cpu_sizes : device_sizes;
    dci.output_shape_ = host2device ? device_sizes : cpu_sizes;
  }
  if (SPYRE_LOG_ENABLED("spyre.runtime",
                        torch_spyre::logging::LogLevel::DEBUG)) {
    std::stringstream s;
    dci.exportJson(s);
    SPYRE_RUNTIME_DEBUG() << "DataConversionInfo: " << s.str();
  }
  return dci;
}

// Empty op needs C++ code and cannot be handled by python side fallback
at::Tensor spyre_empty(c10::IntArrayRef size,
                       std::optional<c10::ScalarType> dtype_opt,
                       std::optional<c10::Layout> layout_opt,
                       std::optional<c10::Device> device_opt,
                       std::optional<bool> pin_memory_opt,
                       std::optional<c10::MemoryFormat> memory_format_opt) {
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  SPYRE_RUNTIME_DEBUG() << "shape=" << size << " on Spyre " << device;
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
              "Non strided layout not supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt),
              "Pin memory can only be on CPU");
  TORCH_CHECK(spyre::is_supported_dtype(dtype),
              "Spyre backend does not support dtype ", dtype);
  const auto memory_format =
      memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(memory_format == c10::MemoryFormat::Contiguous ||
                  memory_format == c10::MemoryFormat::Preserve,
              "Spyre backend only supports contiguous memory format, got: ",
              memory_format);
  const c10::DeviceGuard device_guard(device);

  auto device_layout = SpyreTensorLayout(size.vec(), dtype);
  size_t device_size_bytes = get_device_size_in_bytes(device_layout);
  int64_t cpu_numel = std::accumulate(size.begin(), size.end(), 1LL,
                                      std::multiplies<int64_t>());
  size_t cpu_size_bytes = cpu_numel * c10::elementSize(dtype);
  size_t size_bytes = std::max(device_size_bytes, cpu_size_bytes);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      c10::Storage(c10::make_intrusive<SpyreStorageImpl>(
          c10::StorageImpl::use_byte_size_t(), size_bytes,
          &SpyreAllocator::instance(),
          /*resizeable=*/true)),
      pu1_dks, c10::scalarTypeToTypeMeta(dtype));

  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  spyre_tensor_impl->set_sizes_contiguous(size);
  spyre_tensor_impl->spyre_layout = device_layout;
  SPYRE_RUNTIME_DEBUG() << "SpyreTensorLayout: " << device_layout.toString();
  return tensor;
}

/**
 * This method will determine the size of the tensor on Spyre, then allocate
 * that space on the Spyre and and set the handle for the tensor to that of the
 * memory in the Spyre. For now, it allocates a CPU tensor with the correct
 * size, as the actual storage will stay on CPU until the rest of the stack is
 * ready to filter out the allocation and deallocation of memory from the graph
 * processing.
 */
at::Tensor spyre_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                               std::optional<c10::ScalarType> dtype_opt,
                               std::optional<c10::Layout> layout_opt,
                               std::optional<c10::Device> device_opt,
                               std::optional<bool> pin_memory_opt) {
  // SETUP FOR Spyre TENSOR
  at::detail::check_size_nonnegative(size);
  const auto scalar_type = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(spyre::is_supported_dtype(scalar_type),
              "Spyre backend does not support dtype ", scalar_type);
  caffe2::TypeMeta dtype = c10::scalarTypeToTypeMeta(scalar_type);
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  SPYRE_RUNTIME_DEBUG() << "Tensor info on CPU (Size:" << size
                        << ", Stride: " << stride << ", dtype: " << dtype
                        << ") to be mapped onto device " << device;
  auto device_layout = SpyreTensorLayout(size.vec(), stride.vec(), scalar_type,
                                         generic_stick_dim_order(size.size()));
  size_t device_size_bytes = get_device_size_in_bytes(device_layout);
  int64_t cpu_numel = std::accumulate(size.begin(), size.end(), 1LL,
                                      std::multiplies<int64_t>());
  size_t cpu_size_bytes = cpu_numel * c10::elementSize(scalar_type);
  size_t size_bytes = std::max(device_size_bytes, cpu_size_bytes);

  auto spyre_storage_impl = c10::make_intrusive<SpyreStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes,
      &SpyreAllocator::instance(),
      /*resizeable=*/true);
  auto spyre_storage = c10::Storage(spyre_storage_impl);

  // Create the Spyre Tensor
  const c10::DeviceGuard device_guard(device);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      std::move(spyre_storage), pu1_dks, dtype);

  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  spyre_tensor_impl->set_sizes_and_strides(size, stride);

  spyre_tensor_impl->spyre_layout = device_layout;

  SPYRE_RUNTIME_DEBUG() << "SpyreTensorLayout: " << device_layout.toString();
  return tensor;
}

at::Tensor spyre_empty_with_layout(c10::IntArrayRef size,
                                   c10::IntArrayRef stride,
                                   c10::ScalarType dtype,
                                   SpyreTensorLayout device_layout,
                                   std::optional<c10::Device> device_opt) {
  at::detail::check_size_nonnegative(size);
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  TORCH_CHECK(device.is_privateuseone(),
              "spyre_empty_with_layout expected a Spyre device, got ", device);
  const c10::DeviceGuard device_guard(device);

  size_t device_size_bytes = get_device_size_in_bytes(device_layout);
  int64_t cpu_numel = std::accumulate(size.begin(), size.end(), 1LL,
                                      std::multiplies<int64_t>());
  size_t cpu_size_bytes = cpu_numel * c10::elementSize(dtype);
  size_t size_bytes = std::max(device_size_bytes, cpu_size_bytes);
  auto spyre_storage_impl = c10::make_intrusive<SpyreStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), size_bytes,
      &SpyreAllocator::instance(),
      /*resizeable=*/true);
  auto spyre_storage = c10::Storage(spyre_storage_impl);

  // Create the Spyre Tensor
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      std::move(spyre_storage), pu1_dks, c10::scalarTypeToTypeMeta(dtype));

  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  spyre_tensor_impl->set_sizes_and_strides(size, stride);
  spyre_tensor_impl->spyre_layout = device_layout;
  SPYRE_RUNTIME_DEBUG() << "SpyreTensorLayout: " << device_layout.toString();
  return tensor;
}

at::Tensor& spyre_set_storage(at::Tensor& result, at::Storage storage,
                              int64_t storage_offset, c10::IntArrayRef size,
                              c10::IntArrayRef stride) {
  SPYRE_RUNTIME_DEBUG() << "set method";
  return at::cpu::set_(result, storage, storage_offset, size, stride);
}

/**
 * This method handles copy between devices. When copying to Spyre, this method
 * marks the tensor to compute on Spyre, but continue to use CPU tensor for now
 * such that when we run an op on the tensor on the Spyre, it will have the
 * proper handle to the Spyre allocation
 */
at::Tensor spyre_copy_from(const at::Tensor& self, const at::Tensor& dst,
                           bool non_blocking) {
  SpyreStream stream;
  bool needs_staging = false;
  at::Tensor alloc_view;
  at::Tensor cpu_alloc;

  if (dst.is_privateuseone()) {
    stream = getCurrentStream(dst.device());
  } else {
    stream = getCurrentStream(self.device());
    // D2H staging path: DMA the full physical allocation into a CPU buffer
    // using the original allocation geometry (reconstructed from spyre_layout),
    // then apply the logical view on the CPU side.
    //
    // This path is taken when:
    //   (a) the tensor is expanded/repeated, where we transfer the minimal
    //       amount of data over DMA and allow the CPU to perform the
    //       expand/repeat.
    //   (b) the tensor is sliced along a tiled dimension, starts in the middle
    //       of a tile, and goes beyond the tile it starts in.
    //   (c) the tensor is sliced along a dimension not in the stride_map.
    if (self.is_privateuseone()) {
      auto* spyre_impl =
          static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
      const bool expanded = std::ranges::any_of(
          self.strides(), [](const int64_t& stride) { return stride < 1; });
      const auto [alloc_sizes, alloc_strides] =
          reconstruct_dma_geometry(spyre_impl->spyre_layout);
      const int64_t alloc_numel = c10::multiply_integers(alloc_sizes);
      TORCH_CHECK(alloc_numel >= self.numel(), "Reconstructed DMA allocation (",
                  alloc_numel, " elements) is smaller than tensor numel (",
                  self.numel(), "); layout is inconsistent");
      if (expanded) {
        needs_staging = true;
      } else if (alloc_numel > self.numel()) {
        // alloc_numel == self.numel() (the else branch): tensor fills the
        // allocation exactly; DMA can proceed without staging.
        const auto stl = spyre_impl->spyre_layout;
        const int device_rank = static_cast<int>(stl.stride_map.size());
        const int alloc_rank = static_cast<int>(alloc_strides.size());
        // For each alloc dim j, collect the host strides of device dims that
        // fall within that alloc dim's extent, sorted ascending (innermost
        // first). Used below to detect cross-tile-boundary slices.
        std::vector<std::vector<int64_t>> alloc_tile_dims(alloc_rank);
        for (int j = 0; j < alloc_rank; j++) {
          const int64_t lo = alloc_strides[j];
          const int64_t hi = alloc_sizes[j] * lo;
          for (int k = 0; k < device_rank; k++) {
            const int64_t sm = stl.stride_map[k];
            if (sm >= lo && sm < hi) {
              alloc_tile_dims[j].push_back(sm);
            }
          }
          std::sort(alloc_tile_dims[j].begin(), alloc_tile_dims[j].end());
        }

        // For each logical dim, check whether the slice crosses a tile
        // boundary. A dimension whose stride matches an alloc stride exactly
        // is not tiled and never crosses a boundary. A dimension whose stride
        // falls strictly inside an alloc dim's extent is either a tiled
        // sub-dim (stride present in alloc_tile_dims) or a strided-slice /
        // view of that alloc dim (stride not present). The latter requires
        // staging only if it doesn't cover the full alloc extent.
        const int self_rank = self.dim();
        const int64_t storage_off = self.storage_offset();
        std::vector<bool> is_view(alloc_rank, false);
        std::vector<int64_t> view_sizes(alloc_rank, 1);
        for (int i = 0; i < self_rank && !needs_staging; i++) {
          const int64_t stride = self.strides()[i];
          const int64_t size = self.sizes()[i];
          if (size == 1) continue;
          for (int j = 0; j < alloc_rank; j++) {
            const int64_t alloc_stride = alloc_strides[j];
            const int64_t next_stride = alloc_sizes[j] * alloc_stride;
            if (stride < alloc_stride || stride >= next_stride) continue;
            view_sizes[j] *= size;
            if (stride == alloc_stride) break;  // exact match, no tiling
            // stride falls inside alloc dim j: locate it in alloc_tile_dims[j]
            const auto& tile_dims = alloc_tile_dims[j];
            const auto it =
                std::find(tile_dims.begin(), tile_dims.end(), stride);
            if (it == tile_dims.end()) {
              // Stride not in tile_map: strided-slice or view of
              // alloc_sizes[j]. Staging needed if it doesn't cover the full
              // alloc extent (checked after the loop).
              is_view[j] = true;
              break;
            }
            const size_t index = static_cast<size_t>(it - tile_dims.begin());
            const int64_t next_tile_stride = index + 1 < tile_dims.size()
                                                 ? tile_dims[index + 1]
                                                 : next_stride;
            const int64_t tile_size = next_tile_stride / stride;
            const int64_t dim_offset = (storage_off % next_stride) / stride;
            const int64_t tile_offset = dim_offset % tile_size;
            if (tile_offset != 0 && tile_offset + size > tile_size) {
              needs_staging = true;
            }
            break;
          }
        }
        for (int j = 0; j < alloc_rank && !needs_staging; j++) {
          if (is_view[j] && view_sizes[j] != alloc_sizes[j]) {
            needs_staging = true;
          }
        }
      }

      if (needs_staging) {
        alloc_view = at::as_strided(self, alloc_sizes, alloc_strides,
                                    /*storage_offset=*/0);
        cpu_alloc = at::empty(alloc_sizes, dst.options());
      }
    }
  }

  if (needs_staging) {
    stream.copyAsync(alloc_view, cpu_alloc);
  } else {
    stream.copyAsync(self, dst);
  }
  if (!non_blocking) {
    stream.synchronize();
  }

  if (needs_staging) {
    at::Tensor cpu_view = cpu_alloc.as_strided(self.sizes(), self.strides(),
                                               self.storage_offset());
    dst.copy_(cpu_view);
  }
  return dst;
}

at::Tensor empty_with_layout(
    c10::IntArrayRef size, SpyreTensorLayout device_layout,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Layout> layout_opt,
    std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  c10::Device device = device_opt.value_or(
      c10::impl::VirtualGuardImpl{c10::DeviceType::PrivateUse1}.getDevice());
  SPYRE_RUNTIME_DEBUG() << "shape=" << size << " on Spyre " << device;
  const auto dtype = c10::dtype_or_default(dtype_opt);
  TORCH_CHECK(device.is_privateuseone());
  TORCH_CHECK(c10::layout_or_default(layout_opt) == c10::Layout::Strided,
              "Non strided layout not supported");
  TORCH_CHECK(!c10::pinned_memory_or_default(pin_memory_opt),
              "Pin memory can only be on CPU");
  TORCH_CHECK(spyre::is_supported_dtype(dtype),
              "Spyre backend does not support dtype ", dtype);
  const auto memory_format =
      memory_format_opt.value_or(c10::MemoryFormat::Contiguous);
  TORCH_CHECK(memory_format == c10::MemoryFormat::Contiguous ||
                  memory_format == c10::MemoryFormat::Preserve,
              "Spyre backend only supports contiguous memory format, got: ",
              memory_format);
  const c10::DeviceGuard device_guard(device);

  size_t device_size_bytes = get_device_size_in_bytes(device_layout);
  int64_t cpu_numel = std::accumulate(size.begin(), size.end(), 1LL,
                                      std::multiplies<int64_t>());
  size_t cpu_size_bytes = cpu_numel * c10::elementSize(dtype);
  size_t size_bytes = std::max(device_size_bytes, cpu_size_bytes);
  constexpr c10::DispatchKeySet pu1_dks(c10::DispatchKey::PrivateUse1);
  auto tensor = at::detail::make_tensor_base<SpyreTensorImpl>(
      c10::Storage(c10::make_intrusive<SpyreStorageImpl>(
          c10::StorageImpl::use_byte_size_t(), size_bytes,
          &SpyreAllocator::instance(),
          /*resizeable=*/true)),
      pu1_dks, c10::scalarTypeToTypeMeta(dtype));

  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl());
  spyre_tensor_impl->set_sizes_contiguous(size);
  spyre_tensor_impl->spyre_layout = device_layout;
  SPYRE_RUNTIME_DEBUG() << "SpyreTensorLayout: " << device_layout.toString();
  return tensor;
}

at::Tensor py_empty_with_layout(
    c10::IntArrayRef size, SpyreTensorLayout device_layout,
    std::optional<c10::ScalarType> dtype_opt,
    std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  return empty_with_layout(size, device_layout, dtype_opt,
                           /*layout_opt=*/std::nullopt, device_opt,
                           pin_memory_opt, memory_format_opt);
}

const at::Tensor& spyre_resize_(
    const at::Tensor& self, c10::SymIntArrayRef size,
    std::optional<c10::MemoryFormat> memory_format_opt) {
  auto size_int = c10::asIntArrayRefUnchecked(size);
  // Case 1: No-op.
  if (self.sizes() == size_int && self.is_contiguous()) {
    return self;
  }
  TORCH_CHECK(memory_format_opt != c10::MemoryFormat::Preserve,
              "aten::resize_ does not support MemoryFormat::Preserve");
  TORCH_CHECK(!memory_format_opt.has_value() ||
                  *memory_format_opt == c10::MemoryFormat::Contiguous,
              "aten::resize_ on Spyre only supports contiguous memory format");
  const auto dtype = c10::typeMetaToScalarType(self.dtype());
  TORCH_CHECK(spyre::is_supported_dtype(dtype),
              "Spyre backend does not support dtype ", dtype);

  auto* self_impl = static_cast<SpyreTensorImpl*>(self.unsafeGetTensorImpl());
  // Use STL device bytes (stick-padded) to determine if existing allocation
  // suffices.
  auto new_layout = SpyreTensorLayout(size_int.vec(), dtype);
  const size_t new_device_bytes = get_device_size_in_bytes(new_layout);
  const size_t new_cpu_bytes =
      at::detail::computeStorageNbytesContiguous(size_int, self.itemsize());
  const size_t new_size_bytes = std::max(new_device_bytes, new_cpu_bytes);
  // Case 2: Same-numel or shrink — reinterpret storage in-place, no data moved.
  // Only valid when new last dim ≤ old last dim; otherwise D2H reads into stick
  // padding.
  const int64_t new_numel = c10::multiply_integers(size_int);
  const bool last_dim_ok = size_int.empty() || self.sizes().empty() ||
                           size_int.back() <= self.sizes().back();
  if (new_size_bytes <= self.storage().nbytes() && new_numel <= self.numel() &&
      last_dim_ok) {
    self_impl->set_sizes_contiguous(size_int);
    self_impl->spyre_layout = new_layout;
    SPYRE_RUNTIME_DEBUG() << "to shape=" << size_int
                          << " layout=" << self_impl->spyre_layout.toString();
    return self;
  }
  // Case 3: Reallocate — D2H → CPU resize_ → H2D. Handles expand and any
  // reshape where the new last dim > old last dim (stick-layout incompatible).
  // TODO(kunuruabhishek): avoid round-trip once restickify supports
  // cross-layout D2D copies.
  at::Tensor cpu_buf = self.cpu();
  cpu_buf.resize_(size_int);
  auto new_storage_impl = c10::make_intrusive<SpyreStorageImpl>(
      c10::StorageImpl::use_byte_size_t(), new_size_bytes,
      &SpyreAllocator::instance(), /*resizeable=*/true);
  self_impl->set_storage_keep_dtype(c10::Storage(new_storage_impl));
  self_impl->set_sizes_contiguous(size_int);
  self_impl->spyre_layout = new_layout;
  at::_copy_from(cpu_buf, self, /*non_blocking=*/false);
  SPYRE_RUNTIME_DEBUG() << "expand to shape=" << size_int
                        << " layout=" << self_impl->spyre_layout.toString();
  return self;
}

at::Tensor spyre_fill_tensor(const at::Tensor& self, double value) {
  TORCH_CHECK(self.is_privateuseone(),
              "spyre_fill_tensor: tensor must be on spyre device");
  TORCH_CHECK(self.numel() > 0, "spyre_fill_tensor: cannot fill empty tensor");

  // Map torch dtype to DataFormats for the value->pattern conversion, which
  // fillAsync performs internally.
  DataFormats dtype = get_device_dtype(self.scalar_type());

  // Launch a device-side MEMORY_FILL DMA via the typed fillAsync overload.
  SpyreStream stream;
  stream.fillAsync(get_composite_address(self), value, dtype,
                   /*use_dmai=*/true);

  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("empty.memory_format", TORCH_FN(spyre_empty));
  m.impl("empty_strided", TORCH_FN(spyre_empty_strided));
  m.impl("set_.source_Storage_storage_offset", TORCH_FN(spyre_set_storage));
  m.impl("resize_", TORCH_FN(spyre_resize_));
}

}  // namespace spyre
