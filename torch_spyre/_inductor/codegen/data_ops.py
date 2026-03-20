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

from torch_spyre._inductor.codegen.compute_ops import (
    num_bytes,
    DimInfos,
    get_device_size,
    create_tensor_specific_layouts,
    gen_coord_info_value,
)
from torch_spyre._inductor.constants import (
    INPUT_DIM_LABELS,
    OUTPUT_DIM_LABELS,
)
import math


def generate_slice(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    return {
        "reshape": {
            "numCoresUsed_": 1,
            "dscs_": [],
            "coreIdToDscSchedule": {"0": [[0, -1, 0, 0]]},
            "datadscs_": [
                {
                    "reshape": {
                        "coreIdsUsed_": [0],
                        "dimPool_": ["mb", "out"],
                        "primaryDs_": [{"name_": "pds0", "dimNames": ["mb", "out"]}],
                        "labeledDs_": [
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["mb", "out"],
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": {
                                    "mb": 64,
                                    "out": dimensions[0],
                                },
                                "dimToStickSize_": {"out": 64},
                                "validGap_": {
                                    "mb": [[64, 0]],
                                    "out": [[dimensions[0], 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": f"p{i}",
                                        "dimToSize_": {"mb": 1, "out": 64},
                                        "validGap_": {
                                            "mb": [[1, 0]],
                                            "out": [[64, 0]],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[inputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [0],
                                            },
                                        ],
                                    }
                                    for i in range(dimensions[0] // 64)
                                ],
                                "hbmStartAddress_": pointers[inputs[0]["name"]] // 128,
                            },
                            {
                                "pdsName_": "pds0",
                                "wordLength": 2,
                                "dataformat": "SEN169_FP16",
                                "layoutDimOrder_": ["mb", "out"],
                                "stickDimOrder_": ["out"],
                                "dimToLayoutSize_": {
                                    "mb": 1,
                                    "out": dimensions[0],
                                },
                                "dimToStickSize_": {"out": 64},
                                "validGap_": {
                                    "mb": [[1, 0]],
                                    "out": [[dimensions[0], 0]],
                                },
                                "PieceInfo": [
                                    {
                                        "key_": f"p{i}",
                                        "dimToSize_": {"mb": 1, "out": 64},
                                        "validGap_": {
                                            "mb": [[1, 0]],
                                            "out": [[64, 0]],
                                        },
                                        "PlacementInfo": [
                                            {
                                                "type": "hbm",
                                                "memId": [-1],
                                                "startAddr": [
                                                    pointers[outputs[0]["name"]] // 128
                                                ],
                                            },
                                            {
                                                "type": "lx",
                                                "memId": [0],
                                                "startAddr": [16384],
                                            },
                                        ],
                                    }
                                    for i in range(dimensions[0] // 64)
                                ],
                                "hbmStartAddress_": pointers[outputs[0]["name"]] // 128,
                            },
                        ],
                        "op": {
                            "name": "STCDPOpHBM",
                            "gtrIdsUsed": [],
                            "coreIDtoANInfo": {
                                "0": {
                                    "loopCount": {
                                        "out": dimensions[0] // 64,
                                        "mb": 1,
                                    },
                                    "loopCountL3SU": {},
                                    "addr_info_": {
                                        "l3lu": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 1,
                                                "out": 64,
                                            },
                                        },
                                        "l3su": {
                                            "type_": "stride",
                                            "offset_": {
                                                "mb": 1,
                                                "out": 1,
                                            },
                                        },
                                    },
                                    "inpPieceOrder": [
                                        f"p{i}" for i in range(dimensions[0] // 64)
                                    ],
                                    "outPieceOrder": [
                                        f"p{i}" for i in range(dimensions[0] // 64)
                                    ],
                                }
                            },
                            "numClToUse": 1,
                            "cl0ToLxOffsetLU": 0,
                            "cl0ToLxOffsetSU": 0,
                        },
                    }
                }
            ],
        }
    }


def generate_dldsc(pointers, *, op, dimensions, inputs, outputs, **kwargs):
    tensors = inputs + outputs
    input_dtype = inputs[0]["device_layout"].device_dtype
    data_format = input_dtype

    ndim = len(dimensions)

    cores = 1

    # Get operation dim map from the tensor that represents the operation space
    op_dims_tensor = outputs[0]
    dl = op_dims_tensor["device_layout"]
    dim_map = dl.dim_map[::-1][1:]
    dim_labels = INPUT_DIM_LABELS[: ndim - 1] + OUTPUT_DIM_LABELS[:1]
    dim_splits = [1] * (ndim - 1) + [cores]

    # Obtain (padded) dimensions of the op from a spyre tensor layout
    padded_op_dimensions = [
        get_device_size(host_dim, op_dims_tensor) for host_dim in range(ndim)
    ]

    dim_infos = DimInfos(
        dim_map,
        dim_labels,
        dimensions,
        padded_op_dimensions,
        dim_splits,
    )

    layouts = create_tensor_specific_layouts(
        tensors,
        dim_infos,
        op,
        check_stick_dim=True if op == "ReStickifyOpHBM" else False,
        op_dims_tensor=op_dims_tensor,
    )

    # Compute the stick label from the op tensor.
    op_stick_labels = dim_infos.get_tensor_stick_dim_labels(op_dims_tensor)

    core_id_to_wk_slice = {}
    for i in range(cores):
        core_id_to_wk_slice[str(i)] = {
            str(s): i if s in op_stick_labels else 0 for s in dim_labels
        }

    return {
        op: {
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
                di.label: di.nsplits for di in dim_infos.get_op_infos()
            },
            "coreIdToWkSlice_": core_id_to_wk_slice,
            "coreIdToDscSchedule": {str(c): [[-1, 0, 0, 0]] for c in range(cores)},
            "dscs_": [
                {
                    op: {
                        "numCoresUsed_": cores,
                        "numCoreletsUsed_": 1,
                        "coreIdsUsed_": [c for c in range(cores)],
                        "N_": {
                            "name_": "n",
                            **{
                                di.label + "_": di.padded_size
                                for di in dim_infos.get_op_infos()
                            },  # dim sizes before split
                        },
                        "dataStageParam_": {
                            "0": {
                                "ss_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size
                                        for di in dim_infos.get_op_infos()
                                    },
                                },
                                "el_": {
                                    "name_": "core",
                                    **{
                                        di.label + "_": di.split_size
                                        for di in dim_infos.get_op_infos()
                                    },
                                },
                            }
                        },
                        "primaryDsInfo_": {
                            name: {
                                "layoutDimOrder_": layout_info["layout_order"],
                                "stickDimOrder_": layout_info["stick_dim_order"],
                                "stickSize_": [data_format.elems_per_stick()],
                            }
                            for name, layout_info in layouts.items()
                        },
                        "scheduleTree_": [
                            {
                                "nodeType_": "allocate",
                                "name_": f"allocate-Tensor{i}_{'hbm' if tensor['lx_addr'] is None else 'lx'}",
                                "prev_": "",
                                "ldsIdx_": i,
                                "component_": "hbm"
                                if tensor["lx_addr"] is None
                                else "lx",
                                "layoutDimOrder_": dim_infos.get_tensor_op_layout_order(
                                    tensor, op
                                ),
                                "maxDimSizes_": [-1]
                                * len(dim_infos.get_tensor_op_layout_order(tensor, op)),
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
                                        f"[{c}, 0, 0]": str(
                                            pointers[tensor["name"]]
                                            + c
                                            # calculate the prod of dim sizes
                                            # less significant than chosen split dim i.e. the stick
                                            * math.prod(
                                                dim_infos.get_padded_sizes()[:2]
                                            )
                                            * num_bytes(
                                                tensor["device_layout"].device_dtype
                                            )
                                            // cores
                                        )
                                        if tensor["lx_addr"] is None
                                        else tensor["lx_addr"]
                                        for c in range(cores)
                                    },
                                },
                                "coordinates_": {
                                    "coordInfo": {
                                        di.label: gen_coord_info_value(
                                            size=di.split_size
                                            if (di.scale == 1)
                                            else 1,
                                            nsplits=di.nsplits,
                                            elems_per_stick=tensor[
                                                "device_layout"
                                            ].device_dtype.elems_per_stick(),
                                            is_stick_dim=(di.label in op_stick_labels),
                                            is_stick_reduction=(
                                                di.label in op_stick_labels
                                                and di.scale == -1
                                            ),
                                        )
                                        for di in dim_infos.get_tensor_op_infos(
                                            tensor, op
                                        )
                                    },
                                    "coreIdToWkSlice_": {},
                                },
                            }
                            for i, tensor in enumerate(tensors)
                        ],
                        "labeledDs_": [
                            {
                                "ldsIdx_": i,
                                "dsName_": f"Tensor{i}",
                                "dsType_": tensor["ds_type"],
                                "scale_": [
                                    (
                                        di.scale
                                        # TODO: revisit whether this special case can be removed
                                        #       pending change in deeptools
                                        if not (
                                            di.label in op_stick_labels
                                            and di.scale == -1
                                        )
                                        else -2
                                    )
                                    for di in dim_infos.get_tensor_op_infos(tensor, op)
                                ],
                                "wordLength": num_bytes(
                                    tensor["device_layout"].device_dtype
                                ),
                                "dataFormat_": tensor[
                                    "device_layout"
                                ].device_dtype.name,
                                "memOrg_": {
                                    "hbm": {"isPresent": 1},
                                    "lx": {"isPresent": 1},
                                }
                                if tensor["lx_addr"] is None
                                else {"lx": {"isPresent": 1}},
                            }
                            for i, tensor in enumerate(tensors)
                        ],
                        "constantInfo_": {},
                        "computeOp_": [
                            {
                                "opFuncName": op,
                                "exUnit": "sfp",
                                "attributes_": {
                                    "dataFormat_": data_format.name,
                                    "fidelity_": "regular",
                                },
                                "location": "Inner",
                                "inputLabeledDs": [
                                    f"Tensor{i}-idx{i}" for i in range(len(inputs))
                                ],
                                "outputLabeledDs": [
                                    f"Tensor{i}-idx{i}"
                                    for i in range(len(inputs), len(tensors))
                                ],
                            }
                        ],
                    }
                }
            ],
        }
    }
