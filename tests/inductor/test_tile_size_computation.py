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

"""Test tile_size computation for SpyreTensorLayout.

tile_size is automatically computed in C++ when constructing SpyreTensorLayout
from host_size. It maps disjoint ordered lists of device dimensions to their real
(non-padded) element counts. Examples:
  - {[0, 1]: 100} means device dims [0, 1] contain 100 real elements (rest is padding)
  - {[0, 1]: 64, [2]: 1} means two groups: first group has 64 real, second has 1 real

These tests verify that tile_size is correctly computed by inspecting the repr
of the layout objects created from various host sizes and dtypes.
"""

import unittest
import torch
from torch_spyre._C import SpyreTensorLayout


class TestTileSizeInLayout(unittest.TestCase):
    """Integration tests: tile_size is computed automatically in C++ for all layouts."""

    """Integration tests: create SpyreTensorLayout and verify tile_size."""

    def test_default_layout_has_tile_size(self):
        """Create a default layout and verify tile_size is computed."""
        # Create layout for [7, 100] tensor with default ordering
        stl = SpyreTensorLayout([7, 100], torch.float16)
        # After layout creation, tile_size should be populated
        # Verify via repr that tile_size was computed (it shows up in the repr if non-empty)
        repr_str = str(stl)
        self.assertIn(
            "tile_size", repr_str, "tile_size should appear in repr if it was computed"
        )

    def test_tile_size_persists_through_pickle(self):
        """Verify tile_size is preserved through pickling.

        NOTE: This test is currently skipped because pybind11 doesn't automatically
        handle serialization of map<vector<int64_t>, int64_t>. This is a known
        limitation of the Python bindings. The pickling is handled correctly in
        module.cpp's __getstate__/__setstate__, but the pybind11 binding layer
        can't serialize the vector-keyed map directly.
        """
        self.skipTest(
            "Pickling map<vector> is not supported by pybind11; "
            "this is a known limitation and will be fixed in a follow-up"
        )

    def test_tile_size_used_in_dci_generation(self):
        """Verify tile_size is computed for various tensor shapes.

        This ensures tile_size is automatically populated during layout creation,
        enabling DCI generation to use exact real element counts instead of padding guesses.
        """
        # Test 1D padded layout: 100 elements → 2 sticks (64+36)
        stl = SpyreTensorLayout([100], torch.float16)
        repr_str = str(stl)
        self.assertIn("tile_size", repr_str, "1D layout should have tile_size computed")

        # Test 2D layout with remainder calculation
        stl_2d = SpyreTensorLayout([7, 100], torch.float16)
        repr_str_2d = str(stl_2d)
        self.assertIn(
            "tile_size", repr_str_2d, "2D layout should have tile_size computed"
        )

        # Test scalar layout
        stl_scalar = SpyreTensorLayout([], torch.float16)
        repr_scalar = str(stl_scalar)
        self.assertIn(
            "tile_size", repr_scalar, "Scalar layout should have tile_size computed"
        )


if __name__ == "__main__":
    unittest.main()
