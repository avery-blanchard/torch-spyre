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

#include "spyre_tensor_impl.h"

#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <util/sendefs/dataType.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "logging.h"
#include "types_mapping.h"

namespace spyre {

int64_t elems_per_stick(const DataFormats& df) {
  // TODO(dgrove-oss): DeepTools dataFormatToStickSize map is incomplete!
  auto it = dataFormatToStickSize.find(df);
  if (it != dataFormatToStickSize.end() && it->second > 0) {
    return static_cast<int64_t>(it->second);
  }
  // DCI already accepts these storage formats, missing from the map above.
  // Complete their geometry without enabling unmapped compute formats.
  if (df == DataFormats::IEEE_INT32 || df == DataFormats::IEEE_INT64 ||
      df == DataFormats::BOOL || df == DataFormats::BFLOAT16) {
    return getNumElemsInStick(df);
  }
  // Capability queries use zero for unsupported geometry; storage sizing below
  // rejects it rather than allocating an unknown format.
  return 0;
}

std::vector<int32_t> generic_stick_dim_order(int32_t num_dims) {
  std::vector<int32_t> dim_order;
  for (int32_t i = 0; i < num_dims; i++) {
    dim_order.push_back(i);
  }
  return dim_order;
}

static std::vector<int64_t> compute_host_stride(
    const std::vector<int64_t>& host_size) {
  int n = host_size.size();
  std::vector<int64_t> host_stride(n);
  int64_t stride = 1;
  for (int i = n - 1; i >= 0; --i) {
    host_stride[i] = stride;
    stride *= host_size[i];
  }
  return host_stride;
}

void SpyreTensorLayout::init(std::vector<int64_t> host_size,
                             c10::ScalarType dtype) {
  int host_dims = static_cast<int32_t>(host_size.size());
  auto host_strides = compute_host_stride(host_size);
  auto dim_order = generic_stick_dim_order(host_dims);
  init(host_size, host_strides, dtype, dim_order);
}

void SpyreTensorLayout::init(std::vector<int64_t> host_size,
                             std::vector<int64_t> host_strides,
                             c10::ScalarType dtype,
                             std::vector<int32_t> dim_order) {
  TORCH_CHECK(host_size.size() == host_strides.size(),
              "Incompatible host_size and host_strides");
  TORCH_CHECK((host_size.size() == dim_order.size()) ||
                  (((host_size.size() + 1) == dim_order.size()) &&
                   dim_order.back() == -1),
              "Incompatible host_size and dim_order");

  auto str_type = torchScalarToString[dtype];
  const auto [sen_dtype_cpu, sen_dtype_dev] =
      stringToDTDataFormatPair(str_type);
  this->device_dtype = sen_dtype_dev;

  if (host_size.size() == 0) {
    // Degenerate case of 0-dimension tensor (ie, a scalar)
    this->device_size.resize(2);
    this->device_size[0] = 1;
    this->device_size[1] = this->elems_per_stick();
    this->stride_map.resize(2);
    this->stride_map[0] = -1;
    this->stride_map[1] = -1;
    return;
  }

  // Device layout: dim_order has N entries; the device always has N+1 slots.
  // slot_dim maps each device slot to its host dim: dim_order rotated left by
  // one, with the stick dim (dim_order[N-1]) appended again as the trailing
  // elements-within-stick slot.
  int N = static_cast<int>(dim_order.size());
  int dev_dims = N + 1;
  bool sparse = dim_order.back() == -1;
  int32_t stick_dim = dim_order[N - 1];  // -1 for sparse
  int64_t elems_in_stick = sparse ? 1 : this->elems_per_stick();

  std::vector<int32_t> slot_dim(dim_order.begin() + 1, dim_order.end());
  slot_dim.push_back(dim_order[0]);
  slot_dim.push_back(dim_order[N - 1]);

  // Compute device_size from slot_dim.
  this->device_size.resize(dev_dims);
  this->device_size[N] = this->elems_per_stick();
  for (int i = 0; i < N; ++i) {
    int32_t d = slot_dim[i];
    if (d == stick_dim) {
      this->device_size[i] =
          sparse ? 1 : (host_size[d] + elems_in_stick - 1) / elems_in_stick;
    } else {
      this->device_size[i] = host_size[d];
    }
  }

  // Compute stride_map with a back-to-front pass over slot_dim.  When a host
  // dim appears in two slots (the stick dim), the fine-grained slot (visited
  // last) gets host_strides[d] and the coarser slot inherits the updated value.
  int host_n = static_cast<int>(host_size.size());
  this->stride_map.assign(dev_dims, -1);
  std::vector<int64_t> last_stride(host_n, -1);
  for (int j = dev_dims - 1; j >= 0; --j) {
    int32_t d = slot_dim[j];
    if (d == -1 || host_size[d] == 1) continue;
    this->stride_map[j] =
        last_stride[d] == -1 ? host_strides[d] : last_stride[d];
    last_stride[d] = std::min(this->stride_map[j] * this->device_size[j],
                              host_strides[d] * host_size[d]);
  }
}

std::string SpyreTensorLayout::toString() const {
  std::stringstream ss;
  ss << "SpyreTensorLayout(";
  ss << "device_size=[";
  for (size_t i = 0; i < this->device_size.size(); i++) {
    ss << this->device_size[i];
    if (i < this->device_size.size() - 1) {
      ss << ", ";
    }
  }
  ss << "], stride_map =[";
  for (size_t i = 0; i < this->stride_map.size(); i++) {
    ss << this->stride_map[i];
    if (i < this->stride_map.size() - 1) {
      ss << ", ";
    }
  }
  ss << "], device_dtype=DataFormats.";
  ss << EnumsConversion::dataFormatsToString(this->device_dtype);
  if (this->element_arrangement != ElementArrangement::STANDARD) {
    ss << ", element_arrangement=ElementArrangement.";
    ss << spyre::elementArrangementToString(this->element_arrangement);
  }
  ss << ")";
  return ss.str();
}

SpyreTensorImpl::SpyreTensorImpl(c10::Storage&& storage,
                                 c10::DispatchKeySet key_set,
                                 const caffe2::TypeMeta& dtype)
    : TensorImpl(std::move(storage), key_set, dtype) {
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::Default);
}

SpyreTensorImpl::SpyreTensorImpl(at::TensorImpl::ImplType unused,
                                 c10::Storage&& storage,
                                 c10::DispatchKeySet key_set,
                                 const caffe2::TypeMeta data_type)
    : TensorImpl(unused, std::move(storage), key_set, data_type) {
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::Default);
}

SpyreTensorImpl::SpyreTensorImpl(c10::Storage storage,
                                 c10::DispatchKeySet key_set,
                                 const caffe2::TypeMeta& dtype,
                                 SpyreTensorLayout stl)
    : TensorImpl(std::move(storage), key_set, dtype) {
  set_custom_sizes_strides(c10::TensorImpl::SizesStridesPolicy::Default);
  this->spyre_layout = stl;
}

// FIXME: This is currently returning cpu storage as other methods use it, but
// will return Spyre storage in a later PR
const at::Storage& SpyreTensorImpl::storage() const {
  return storage_;
}

template <typename VariableVersion>
c10::intrusive_ptr<c10::TensorImpl>
SpyreTensorImpl::shallow_copy_and_detach_core(
    const VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  if (key_set_.has(c10::DispatchKey::Python) &&
      !c10::impl::tls_is_dispatch_key_excluded(c10::DispatchKey::Python)) {
    auto r = (*c10::impl::getGlobalPyInterpreter())->detach(this);
    if (r) {
      r->set_version_counter(version_counter);
      r->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      return r;
    }
  }
  auto impl = c10::make_intrusive<SpyreTensorImpl>(storage_, key_set_,
                                                   data_type_, spyre_layout);
  impl->dma_sizes = this->dma_sizes;
  impl->dma_strides = this->dma_strides;
  copy_tensor_metadata(
      /*src_impl=*/this,
      /*dest_impl=*/impl.get(),
      /*version_counter=*/version_counter,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);

  return impl;
}

c10::intrusive_ptr<c10::TensorImpl> SpyreTensorImpl::shallow_copy_and_detach(
    const c10::VariableVersion& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(version_counter,
                                      allow_tensor_metadata_change);
}

at::intrusive_ptr<c10::TensorImpl> SpyreTensorImpl::shallow_copy_and_detach(
    c10::VariableVersion&& version_counter,
    bool allow_tensor_metadata_change) const {
  return shallow_copy_and_detach_core(std::move(version_counter),
                                      allow_tensor_metadata_change);
}

// FIXME: This is a temporary implementation to get the Spyre Tensor with CPU
// storage basic operation (view) to work
void SpyreTensorImpl::shallow_copy_from(
    const at::intrusive_ptr<at::TensorImpl>& impl) {
  auto spyre_impl = static_cast<SpyreTensorImpl*>(impl.get());
  at::TensorImpl::shallow_copy_from(impl);
  this->dma_sizes = spyre_impl->dma_sizes;
  this->dma_strides = spyre_impl->dma_strides;
  this->spyre_layout = spyre_impl->spyre_layout;
}

uint64_t get_device_size_in_bytes(const SpyreTensorLayout& stl) {
  return get_device_size_in_bytes(stl.device_size, stl.device_dtype);
}

uint64_t get_device_size_in_bytes(const std::vector<int64_t>& device_size,
                                  const DataFormats& device_dtype) {
  // Size complete device sticks, including padding and sparse positions. The
  // trailing extent may expose fewer values, but does not shorten a stick.
  // A compute format without storage geometry must not be sized by bit width.
  const auto elems = elems_per_stick(device_dtype);
  TORCH_CHECK(elems > 0, "No device stick geometry for data format ",
              static_cast<int>(device_dtype));
  const auto bits = getSizeInBits(device_dtype);
  TORCH_CHECK(bits > 0, "No device element width for data format ",
              static_cast<int>(device_dtype));
  uint64_t size_bytes = (elems * bits + 7) / 8;
  for (int i = static_cast<int>(device_size.size()) - 2; i >= 0; i--) {
    size_bytes *= device_size[i];
  }
  return size_bytes;
}
SpyreTensorLayout get_spyre_tensor_layout(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone());
  SpyreTensorLayout stl;
  SpyreTensorImpl* impl;
  if (impl = dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl())) {
    stl = impl->spyre_layout;
  } else {
    TORCH_CHECK(false, "Error: Device tensor does not have SpyreTensorLayout");
  }
  return stl;
}

void set_spyre_tensor_layout(const at::Tensor& tensor,
                             const SpyreTensorLayout& stl) {
  TORCH_CHECK(tensor.is_privateuseone());
  SpyreTensorImpl* impl;
  if (impl = dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl())) {
    impl->spyre_layout = stl;
  } else {
    TORCH_CHECK(false,
                "Error: Attempting to set a STL for a device tensor that does "
                "not have SpyreTensorImpl");
  }
}

std::vector<int64_t> get_spyre_tensor_sizes(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone());
  SpyreTensorImpl* impl;
  if (impl = dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl())) {
    return impl->dma_sizes;
  }
  TORCH_CHECK(false, "Error: Device tensor does not have SpyreTensorImpl");
}

std::vector<int64_t> get_spyre_tensor_strides(const at::Tensor& tensor) {
  TORCH_CHECK(tensor.is_privateuseone());
  SpyreTensorImpl* impl;
  if (impl = dynamic_cast<SpyreTensorImpl*>(tensor.unsafeGetTensorImpl())) {
    return impl->dma_strides;
  }
  TORCH_CHECK(false, "Error: Device tensor does not have SpyreTensorImpl");
}

};  // namespace spyre
