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
