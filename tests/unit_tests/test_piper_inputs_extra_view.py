# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for Piper OpenPI inputs when ``extra_view_image`` stacks two wrist cameras."""

import numpy as np

from rlinf.models.embodiment.openpi.policies.piper_policy import _right_wrist_from_extra


def test_right_wrist_from_extra_stack_v_hwc():
    ex = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    ex[1, :, :, :] = 200
    img, ok = _right_wrist_from_extra(ex)
    assert ok
    assert img.shape == (4, 4, 3)
    assert int(img[0, 0, 0]) == 200


def test_right_wrist_from_extra_legacy_h_v_wc():
    ex = np.zeros((4, 2, 4, 3), dtype=np.uint8)
    ex[:, 1, :, :] = 180
    img, ok = _right_wrist_from_extra(ex)
    assert ok
    assert img.shape == (4, 4, 3)
    assert int(img[0, 0, 0]) == 180


def test_right_wrist_from_extra_single_view_returns_false():
    ex = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    _img, ok = _right_wrist_from_extra(ex)
    assert not ok


def test_right_wrist_from_extra_v_chw():
    ex = np.zeros((2, 3, 4, 4), dtype=np.uint8)
    ex[1, 0, :, :] = 99
    img, ok = _right_wrist_from_extra(ex)
    assert ok
    assert img.shape == (4, 4, 3)
    assert int(img[0, 0, 0]) == 99
