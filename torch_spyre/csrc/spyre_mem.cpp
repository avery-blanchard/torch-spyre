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
#include <functional>
#include <map>
#include <memory>
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

/* Valid (non-padded) element count for device dimension dim_idx.
 * If tile_size is populated (from init() path), read directly from it.
 * Otherwise (raw constructor path), default to device_size (assume no padding).
 */
static int64_t valid_size(int64_t dim_idx, const SpyreTensorLayout& stl) {
  // If tile_size is empty, default to device capacity (no padding assumed)
  if (stl.tile_size.empty()) {
    return stl.device_size[dim_idx];
  }

  for (const auto& [dims, valid_count] : stl.tile_size) {
    if (dims.size() == 1) {
      if (dims[0] == dim_idx) return valid_count;
      continue;
    }
    // A multi-key entry is always a stick-count dim paired with its
    // trailing within-stick dim (see SpyreTensorLayout::tile_size doc): the
    // within-stick dimension is sized up to a full stick, and the count
    // dimension gets what remains after dividing out that stick size.
    const int64_t stick_dim = dims.back();
    if (stick_dim != dim_idx && dims.front() != dim_idx) continue;
    const int64_t stick_size = stl.device_size[stick_dim];
    TORCH_CHECK(stick_size > 0, "Invalid device size ", stick_size,
                " for stick dimension ", stick_dim);
    // For stick dim: return min of valid_count and stick_size (saturate at
    // stick boundary) For count dim: return ceiling division (number of sticks
    // needed for valid_count elements)
    return dim_idx == stick_dim ? std::min<int64_t>(valid_count, stick_size)
                                : (valid_count + stick_size - 1) / stick_size;
  }
  return stl.device_size[dim_idx];
}

/* Parallel tile_size-centric implementation of DCI generation.
 *
 * This function generates DataConversionStrideInfo by iterating over
 * tile_size groups (device dim groups) directly instead of deriving
 * dim_map. Each tile_size group maps to a real element count.
 *
 * @param host_strides: Host memory strides for each device dimension
 */
auto get_device_stride_infos_from_tile_size(
    c10::IntArrayRef sizes, c10::IntArrayRef strides, int64_t cpu_offset,
    int64_t device_offset, SpyreTensorLayout stl,
    const std::vector<int64_t>& host_strides, bool host2device)
    -> std::vector<DataConversionStrideInfo> {
  const int device_rank = stl.stride_map.size();

  // Build contiguous device_strides
  std::vector<int64_t> device_strides(device_rank, 1);
  int64_t prev_size = 1;
  for (int i = device_rank - 1; i >= 0; i--) {
    device_strides[i] = prev_size;
    prev_size *= stl.device_size[i];
  }

  // Build dcsi_sizes (unpadded) from tile_size
  std::vector<int64_t> dcsi_sizes(device_rank, 1);

  std::vector<std::vector<int64_t>> remainders;
  std::vector<int64_t> host_offsets;
  std::vector<int64_t> device_offsets;
  remainders.reserve(stl.tile_size.size());

  // Single pass: compute dcsi_sizes and detect remainders
  for (const auto& [dev_dims, num_valid_elems] : stl.tile_size) {
    if (dev_dims.size() == 1) {
      dcsi_sizes[dev_dims[0]] = num_valid_elems;
    } else {
      // Multi-dim (stick) group: compute sizes and check for padding
      int last_dev_dim = dev_dims.back();
      int count_dim = dev_dims[0];
      int64_t elems_per_stick = stl.device_size[last_dev_dim];
      int64_t complete_sticks = num_valid_elems / elems_per_stick;
      int64_t partial_stick_count = num_valid_elems % elems_per_stick;

      // Main transfer: complete sticks only (remainders in separate DCSI)
      int64_t main_stick_count = complete_sticks;
      int64_t main_last_dim_size = elems_per_stick;

      for (int d : dev_dims) {
        dcsi_sizes[d] =
            (d == last_dev_dim) ? main_last_dim_size : main_stick_count;
      }

      // Generate remainder if there's a partial stick
      if (partial_stick_count != 0) {
        std::vector<int64_t> remainder(device_rank, 0);
        remainder[count_dim] = 1;                       // Just the last stick
        remainder[last_dev_dim] = partial_stick_count;  // Partial elements
        remainders.push_back(remainder);
        // Offset moves through the count dimension (sticks)
        host_offsets.push_back(complete_sticks * host_strides[count_dim]);
        device_offsets.push_back(complete_sticks * device_strides[count_dim]);
      }
    }
  }

  // Create first DataConversionStrideInfo (in pre-reversal space)
  DataConversionStrideInfo stride_info;
  stride_info.size_ = dcsi_sizes;
  stride_info.stride_src_ = host2device ? host_strides : device_strides;
  stride_info.stride_dst_ = host2device ? device_strides : host_strides;
  stride_info.offset_src_ = host2device ? cpu_offset : device_offset;
  stride_info.offset_dst_ = host2device ? device_offset : cpu_offset;

  // Build all remainder entries in pre-reversal space
  std::vector<DataConversionStrideInfo> stride_infos = {stride_info};
  for (size_t i = 0; i < remainders.size(); i++) {
    DataConversionStrideInfo info;
    info.size_ = stride_info.size_;
    // Update size_ for remainder: copy non-zero entries from remainders[i]
    for (int k = 0; k < device_rank; k++) {
      if (remainders[i][k] > 0) {
        info.size_[k] = remainders[i][k];
      }
    }
    info.stride_src_ = stride_info.stride_src_;
    info.stride_dst_ = stride_info.stride_dst_;
    info.offset_src_ = stride_info.offset_src_ +
                       (host2device ? host_offsets[i] : device_offsets[i]);
    info.offset_dst_ = stride_info.offset_dst_ +
                       (host2device ? device_offsets[i] : host_offsets[i]);
    stride_infos.push_back(info);
  }

  // Reverse all stride_infos to hardware order (innermost-first)
  for (auto& info : stride_infos) {
    std::reverse(info.size_.begin(), info.size_.end());
    std::reverse(info.stride_src_.begin(), info.stride_src_.end());
    std::reverse(info.stride_dst_.begin(), info.stride_dst_.end());
  }

  return stride_infos;
}

// Rebuild stride_map using host strides instead of device strides.
// The original stride_map was built from device_strides. Use device_strides as
// a guide to determine which host strides should map to each device dimension.
// Rebuild stride_map by replacing device strides with cpu strides.
// stride_map was created using device_strides as host stride values.
// We replace each device_stride value with its corresponding cpu_stride.
// Rebuild stride_map by matching host dims via sizes and replacing strides.
// tile_size groups device dims. For each group, find the corresponding host dim
// by matching sizes, then replace the stride_map entries with cpu_strides.
static std::vector<int64_t> rebuild_stride_map_for_host_strides(
    const std::vector<int64_t>& cpu_sizes,
    const std::vector<int64_t>& dev_sizes,
    const std::vector<int64_t>& dev_strides,
    const std::vector<int64_t>& cpu_strides,
    const std::map<std::vector<int64_t>, int64_t>& tile_size,
    const std::vector<int64_t>& original_stride_map) {
  std::vector<int64_t> new_stride_map = original_stride_map;

  // For each tile_size group (which represents one host dim), find the
  // corresponding host stride in dev_strides and replace with cpu_strides.
  for (const auto& [dev_dims, _] : tile_size) {
    if (dev_dims.empty()) continue;

    // Find the host stride value for this group from stride_map
    int64_t dev_stride_value = 0;
    for (int d : dev_dims) {
      if (original_stride_map[d] > 0) {
        dev_stride_value = original_stride_map[d];
        break;
      }
    }

    if (dev_stride_value <= 0) continue;  // Skip broadcast/sparse dims

    // Match host dimension by finding which host index has dev_stride_value
    // in dev_strides
    int64_t host_dim_idx = -1;
    for (size_t h = 0; h < dev_strides.size(); ++h) {
      if (dev_strides[h] == dev_stride_value) {
        host_dim_idx = h;
        break;
      }
    }

    if (host_dim_idx < 0 ||
        host_dim_idx >= static_cast<int64_t>(cpu_strides.size())) {
      continue;
    }

    // Get the cpu stride for this host dimension and update stride_map
    int64_t cpu_stride_value = cpu_strides[host_dim_idx];
    for (int d : dev_dims) {
      if (original_stride_map[d] == dev_stride_value) {
        new_stride_map[d] = cpu_stride_value;
      }
    }
  }

  return new_stride_map;
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

  auto spyre_tensor_impl =
      static_cast<SpyreTensorImpl*>(dev_tensor->unsafeGetTensorImpl());

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

  const int device_rank = stl.stride_map.size();

  // Compute the device_offset, which is the offset into stl.device_size, based
  // on the dev_offset, which is the offset into dma_sizes.
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

  // Detect slicing via tile_size: compare total valid elements against tensor
  // numel
  int64_t total_valid = 0;
  for (const auto& [dims, count] : stl.tile_size) {
    total_valid += count;
  }
  const bool dev_sliced = total_valid > dev_tensor->numel();

  // For DCI generation, use stride_map (device dims to host strides).
  // When host and device strides differ (H2D transfer), rebuild to use actual
  // host strides instead of the device layout's assumed strides.
  std::vector<int64_t> host_strides_for_dci = stl.stride_map;
  if (host2device && cpu_strides != dev_strides) {
    host_strides_for_dci = rebuild_stride_map_for_host_strides(
        cpu_sizes, dev_sizes, dev_strides, cpu_strides, stl.tile_size,
        stl.stride_map);
  }

  // Use tile_size-centric path for DCI generation
  TORCH_CHECK(
      !stl.tile_size.empty(),
      "SpyreTensorLayout must have tile_size populated for DCI generation");
  dci.dcsi_ = get_device_stride_infos_from_tile_size(
      cpu_sizes, cpu_strides, cpu_offset, device_offset, stl,
      host_strides_for_dci, host2device);

  // Reverse PyTorch ordering
  std::reverse(cpu_sizes.begin(), cpu_sizes.end());
  std::reverse(device_sizes.begin(), device_sizes.end());
  dci.input_shape_ = host2device ? cpu_sizes : device_sizes;
  dci.output_shape_ = host2device ? device_sizes : cpu_sizes;
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
  at::Tensor alloc_view;
  at::Tensor cpu_alloc;
  const at::Tensor* copy_from = &self;
  const at::Tensor* copy_to = &dst;
  bool non_overlapping_and_dense = true;

  if (dst.is_privateuseone()) {
    stream = getCurrentStream(dst.device());
  } else {
    stream = getCurrentStream(self.device());
    // D2H staging path: DMA the full physical allocation into a CPU buffer
    // using dma_sizes/dma_strides/spyre_layout (the layout the data was
    // written with), then apply the logical view on the CPU side.
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
      auto stl = spyre_impl->spyre_layout;

      // Detect expansion or slicing via tile_size
      int64_t total_valid = 0;
      if (!stl.tile_size.empty()) {
        for (const auto& [dims, count] : stl.tile_size) {
          total_valid += count;
        }
      } else {
        total_valid = c10::multiply_integers(stl.device_size);
      }

      if (expanded || total_valid > self.numel()) {
        // Tensor is expanded or sliced: stage through full allocation.
        // Use device_size with contiguous strides (device layout is
        // contiguous).
        non_overlapping_and_dense = false;
        std::vector<int64_t> alloc_sizes = stl.device_size;
        // Compute contiguous strides for device_size
        std::vector<int64_t> alloc_strides(alloc_sizes.size(), 1);
        int64_t stride = 1;
        for (int i = static_cast<int>(alloc_sizes.size()) - 1; i >= 0; --i) {
          alloc_strides[i] = stride;
          stride *= alloc_sizes[i];
        }
        alloc_view = at::as_strided(self, alloc_sizes, alloc_strides,
                                    /*storage_offset=*/0);
        cpu_alloc = at::empty(alloc_sizes, dst.options());
        copy_from = &alloc_view;
        copy_to = &cpu_alloc;
      } else if (total_valid < self.numel() && !stl.tile_size.empty()) {
        // Tensor claims more elements than tile_size allows: stage through full
        // allocation
        non_overlapping_and_dense = false;
        std::vector<int64_t> alloc_sizes = stl.device_size;
        // Compute contiguous strides for device_size
        std::vector<int64_t> alloc_strides(alloc_sizes.size(), 1);
        int64_t stride = 1;
        for (int i = static_cast<int>(alloc_sizes.size()) - 1; i >= 0; --i) {
          alloc_strides[i] = stride;
          stride *= alloc_sizes[i];
        }
        alloc_view = at::as_strided(self, alloc_sizes, alloc_strides,
                                    /*storage_offset=*/0);
        cpu_alloc = at::empty(alloc_sizes, dst.options());
        copy_from = &alloc_view;
        copy_to = &cpu_alloc;
      }
    }
  }

  stream.copyAsync(*copy_from, *copy_to);
  if (!non_blocking) {
    stream.synchronize();
  }

  if (!non_overlapping_and_dense) {
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
    self_impl->dma_sizes = size_int.vec();
    self_impl->dma_strides = self_impl->strides().vec();
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
