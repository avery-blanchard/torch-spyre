# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch_spyre._C import encode_constant, DataFormats


def swap_last_two_elements(x: list):
    assert len(x) >= 2
    return x[:-2] + x[-1:] + x[-2:-1]


def calculate_core_to_slice_mapping(
    iteration_space, dim_splits: list[int]
) -> dict[str, dict[str, int]]:
    """
    Calculate mapping from core ID to slice indices for each dimension.

    Iterates dimensions right-to-left (innermost varies fastest), similar to
    row-major ordering in multi-dimensional arrays.

    Args:
        dim_labels: List of dimension labels (e.g., ["mb", "out", "x"])
        dim_splits: Number of splits per dimension (e.g., [2, 4, 1])

    Returns:
        Dictionary mapping core ID (as string) to dimension slice indices
    """
    total_cores = 1
    for splits in dim_splits:
        total_cores *= splits

    core_to_slice = {}

    for core_id in range(total_cores):
        # Calculate multi-dimensional index from flat core_id
        # Iterate right-to-left (innermost dimension varies fastest)
        indices = {}
        remaining = core_id

        for i, (dim, _) in enumerate(iteration_space.items()):
            indices[str(dim)] = remaining % dim_splits[i]
            remaining //= dim_splits[i]

        core_to_slice[str(core_id)] = indices

    return core_to_slice


# def core_idx_to_slice_offset(
#     dim_info_list: list[DimInfo],
#     wk_slice: dict[str, int],
#     device_size: list[int],
# ) -> int:
#     # compute tensor specific strides from its device layout
#     strides = {}
#     for i, di in enumerate(dim_info_list):
#         strides[di.label] = math.prod(device_size[-i - 2 :])

#     # Calculate offset by accumulating contribution from each dimension
#     offset = 0
#     for di in dim_info_list:
#         label = di.label
#         slice_idx = wk_slice[label]
#         offset += slice_idx * strides[label] // di.nsplits

#     return offset


def num_bytes(df: DataFormats) -> int:
    """Try to avoid using this method; it is a bad API due to sub-byte datatypes"""
    num_elems = df.elems_per_stick()
    if num_elems > 128:
        raise RuntimeError(f"sub-byte dataformat {df}")
    return 128 // num_elems


def generate_constant_info(data_format, constants):
    if len(constants.keys()) == 0:
        return "{}"
    constant_info = {}
    for name, value in constants.items():
        ci = {
            "dataFormat_": data_format.name,
            "name_": name,
            "data_": {
                "dim_prop_func": [{"Const": {}}, {"Const": {}}, {"Map": {}}],
                "dim_prop_attr": [
                    {"factor_": 1, "label_": "core"},
                    {"factor_": 1, "label_": "corelet"},
                    {"factor_": 1, "label_": "time"},
                ],
                "data_": {"[0, 0, 0]": [encode_constant(value, data_format)]},
            },
        }
        constant_info[f"{len(constant_info)}"] = ci
    return constant_info


def add_constant(kwargs, name, value) -> int:
    """
    Add a constant to kwargs['op_info']['constants'] and return its index.
    Returns:
        int: The index of the newly added constant (0-based)
    """
    # Ensure structure exists
    if "op_info" not in kwargs:
        kwargs["op_info"] = {}
    if "constants" not in kwargs["op_info"]:
        kwargs["op_info"]["constants"] = {}

    index = len(kwargs["op_info"]["constants"])
    kwargs["op_info"]["constants"][name] = value

    return index


def gen_coord_info_value(
    size: int,
    nsplits: int,
    elems_per_stick: int,
    is_stick_dim: bool,
    is_stick_reduction: bool = False,
):
    return (
        {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 1,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {
                        "Affine": {
                            "alpha_": size,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 1,
                            "beta_": 0,
                        }
                    },
                ],
                "dim_prop_attr": [
                    {
                        "factor_": nsplits,
                        "label_": "core_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "corelet_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "row_fold",
                    },
                    {
                        "factor_": size,
                        "label_": "elem_arr_0",
                    },
                ],
            },
        }
        if not is_stick_dim
        else {
            "spatial": 3,
            "temporal": 0,
            "elemArr": 2,
            "padding": "nopad",
            "folds": {
                "dim_prop_func": [
                    {
                        "Affine": {
                            "alpha_": elems_per_stick if is_stick_reduction else size,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": elems_per_stick,
                            "beta_": 0,
                        }
                    },
                    {
                        "Affine": {
                            "alpha_": 0 if is_stick_reduction else 1,
                            "beta_": 0,
                        }
                    },
                ],
                "dim_prop_attr": [
                    {
                        "factor_": nsplits,
                        "label_": "core_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "corelet_fold",
                    },
                    {
                        "factor_": 1,
                        "label_": "row_fold",
                    },
                    {
                        "factor_": 1
                        if is_stick_reduction
                        else (size // elems_per_stick),
                        "label_": "elem_arr_1",
                    },
                    {
                        "factor_": elems_per_stick,
                        "label_": "elem_arr_0",
                    },
                ],
            },
        }
    )


def generate_sdsc(sdsc_spec):
    ndim = len(sdsc_spec.iteration_space)
    cores = 1
    dim_splits = [1] * ndim
    out_idx = len(sdsc_spec.args) - 1
    core_id_to_wk_slice = calculate_core_to_slice_mapping(
        sdsc_spec.iteration_space, dim_splits
    )
    return {
        sdsc_spec.opfunc: {
            "sdscFoldProps_": [{"factor_": 1, "label_": "time"}],
            "sdscFolds_": {
                "dim_prop_func": [{"Affine": {"alpha_": 1, "beta_": 0}}],
                "dim_prop_attr": [{"factor_": 1, "label_": "time"}],
                "data_": {"[0]": "0"},
            },
            "coreFoldProp_": {"factor_": cores, "label_": "core"},
            "coreletFoldProp_": {"factor_": 1, "label_": "corelet"},
            "numCoresUsed_": cores,
            "coreIdToDsc_": {str(c): 0 for c in range(cores)},
            "numWkSlicesPerDim_": {
                str(dim): num_wk_slices
                for dim, num_wk_slices in sdsc_spec.work_slices.items()
            },
            "coreIdToWkSlice_": core_id_to_wk_slice,
            "coreIdToDscSchedule": {str(c): [[-1, 0, 0, 0]] for c in range(cores)},
            "dscs_": [
                {
                    sdsc_spec.opfunc: {
                        "numCoresUsed_": cores,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [c for c in range(cores)],
                        "N_": {
                            "name_": "n",
                            **{
                                str(dim) + "_": size
                                for dim, size in sdsc_spec.iteration_space.items()
                            },  # dim sizes before split
                        },
                        "coordinateMasking_": {
                            str(dim): mask_range
                            for dim, mask_range in sdsc_spec.coordinate_masking.items()
                        },
                        "maskingConstId_": 0 if sdsc_spec.coordinate_masking else -1,
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        str(dim) + "_": size
                                        for dim, size in sdsc_spec.iteration_space.items()
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        str(dim) + "_": size
                                        for dim, size in sdsc_spec.iteration_space.items()
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            label: {
                                "layoutDimOrder_": [
                                    str(dim) for dim in layout_info["dim_order"]
                                ],
                                "stickDimOrder_": [str(layout_info["stick_dim_order"])],
                                "stickSize_": [layout_info["stick_size"]],
                            }
                            for label, layout_info in sdsc_spec.layouts.items()
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": f"allocate-Tensor{i}_{'hbm'}",  # TODO(avery)
                                "prev_": "",
                                "ldsIdx_": i,
                                "component_": "hbm",  # TODO(avery)
                                "layoutDimOrder_": [
                                    str(dim)
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                "maxDimSizes_": [
                                    tensor.max_dim_sizes[dim]
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                "startAddressCoreCorelet_": {
                                    "dim_prop_func": [
                                        {"Map": {}},
                                        {"Const": {}},
                                        {"Const": {}},
                                    ],
                                    "dim_prop_attr": [
                                        {"factor_": cores, "label_": "core"},
                                        {"factor_": 1, "label_": "corelet"},
                                        {"factor_": 1, "label_": "time"},
                                    ],
                                    "data_": {
                                        f"[{c}, 0, 0]": str(tensor.start_address)
                                        for c in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        str(dim): gen_coord_info_value(
                                            size=sdsc_spec.iteration_space[dim]
                                            if (tensor.scales[dim] == 1)
                                            else 1,
                                            nsplits=1,
                                            elems_per_stick=tensor.data_format.elems_per_stick(),
                                            is_stick_dim=(
                                                sdsc_spec.layouts[tensor.layout][
                                                    "stick_dim_order"
                                                ].has(dim)
                                            ),
                                            is_stick_reduction=(
                                                tensor.scales[dim] == -2
                                            ),
                                        )
                                        for dim in sdsc_spec.layouts[tensor.layout][
                                            "dim_order"
                                        ]
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for i, tensor in enumerate(sdsc_spec.args)
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": i,
                                "dsName_": f"Tensor{i}",
                                "dsType_": tensor.layout,
                                "scale_": [
                                    tensor.scales[dim]
                                    for dim in sdsc_spec.layouts[tensor.layout][
                                        "dim_order"
                                    ]
                                ],
                                "wordLength": num_bytes(tensor.data_format),
                                "dataFormat_": tensor.data_format.name,
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                },
                                # if tensor["lx_addr"] is None
                                # else {"lx": {"isPresent": 1}},
                            }
                            for i, tensor in enumerate(sdsc_spec.args)
                        ],
                        "constantInfo_": generate_constant_info(
                            sdsc_spec.data_format, sdsc_spec.constants
                        ),
                        "computeOp_": [
                            {
                                "exUnit": sdsc_spec.execution_unit,
                                "opFuncName": sdsc_spec.opfunc,
                                "attributes_": {
                                    "dataFormat_": sdsc_spec.data_format.name,
                                    "fidelity_": "regular",
                                },
                                "location": "Inner",
                                "inputLabeledDs": [
                                    f"Tensor{i}-idx{i}"
                                    for i in range(sdsc_spec.num_inputs)
                                ],
                                "outputLabeledDs": [f"Tensor{out_idx}-idx{out_idx}"],
                            }
                        ],
                    }
                }
            ],
        }
    }
