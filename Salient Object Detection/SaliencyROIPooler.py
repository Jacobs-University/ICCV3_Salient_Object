#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import math
import sys
import torch
from torch import nn
from torchvision.ops import RoIPool

from detectron2.layers import ROIAlign, ROIAlignRotated, cat
torch.cuda.empty_cache()
__all__ = ["ROIPooler"]


def assign_boxes_to_levels(box_lists, min_level, max_level, canonical_box_size, canonical_level):

    eps = sys.float_info.epsilon
    box_sizes = torch.sqrt(cat([boxes.area() for boxes in box_lists]))
    # Eqn.(1) in FPN paper
    level_assignments = torch.floor(
        canonical_level + torch.log2(box_sizes / canonical_box_size + eps)
    )
    # clamp level to (min, max), in case the box size is too large or too small
    # for the available feature maps
    level_assignments = torch.clamp(level_assignments, min=min_level, max=max_level)
    return level_assignments.to(torch.int64) - min_level


def convert_boxes_to_pooler_format(box_lists):

    def fmt_box_list(box_tensor, batch_index):
        repeated_index = torch.full(
            (len(box_tensor), 1), batch_index, dtype=box_tensor.dtype, device=box_tensor.device
        )
        return cat((repeated_index, box_tensor), dim=1)

    pooler_fmt_boxes = cat(
        [fmt_box_list(box_list.tensor, i) for i, box_list in enumerate(box_lists)], dim=0
    )

    return pooler_fmt_boxes


class ROIPooler(nn.Module):

    def __init__(
        self,
        output_size,
        scales,
        sampling_ratio,
        pooler_type,
        canonical_box_size=224,
        canonical_level=4,
    ):

        super().__init__()

        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert len(output_size) == 2
        assert isinstance(output_size[0], int) and isinstance(output_size[1], int)
        self.output_size = output_size

        if pooler_type == "ROIAlign":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=False
                )
                for scale in scales
            )
        elif pooler_type == "ROIAlignV2":
            self.level_poolers = nn.ModuleList(
                ROIAlign(
                    output_size, spatial_scale=scale, sampling_ratio=sampling_ratio, aligned=True
                )
                for scale in scales
            )
        elif pooler_type == "ROIPool":
            self.level_poolers = nn.ModuleList(
                RoIPool(output_size, spatial_scale=scale) for scale in scales
            )
        elif pooler_type == "ROIAlignRotated":
            self.level_poolers = nn.ModuleList(
                ROIAlignRotated(output_size, spatial_scale=scale, sampling_ratio=sampling_ratio)
                for scale in scales
            )
        else:
            raise ValueError("Unknown pooler type: {}".format(pooler_type))

        # Map scale (defined as 1 / stride) to its feature map level under the
        # assumption that stride is a power of 2.
        min_level = -math.log2(scales[0])
        max_level = -math.log2(scales[-1])
        assert math.isclose(min_level, int(min_level)) and math.isclose(
            max_level, int(max_level)
        ), "Featuremap stride is not power of 2!"
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        assert (
            len(scales) == self.max_level - self.min_level + 1
        ), "[ROIPooler] Sizes of input featuremaps do not form a pyramid!"
        assert 0 < self.min_level and self.min_level <= self.max_level
        if len(scales) > 1:
            # When there is only one feature map, canonical_level is redundant and we should not
            # require it to be a sensible value. Therefore we skip this assertion
            assert self.min_level <= canonical_level and canonical_level <= self.max_level
        self.canonical_level = canonical_level
        assert canonical_box_size > 0
        self.canonical_box_size = canonical_box_size

    def forward(self, x, box_lists):
 
        num_level_assignments = len(self.level_poolers)

        assert isinstance(x, list) and isinstance(
            box_lists, list
        ), "Arguments to pooler must be lists"
        assert (
            len(x) == num_level_assignments
        ), "unequal value, num_level_assignments={}, but x is list of {} Tensors".format(
            num_level_assignments, len(x)
        )

        assert len(box_lists) == x[0].size(
            0
        ), "unequal value, x[0] batch dim 0 is {}, but box_list has length {}".format(
            x[0].size(0), len(box_lists)
        )

        pooler_fmt_boxes = convert_boxes_to_pooler_format(box_lists)

        if num_level_assignments == 1:
            return self.level_poolers[0](x[0], pooler_fmt_boxes)

        level_assignments = assign_boxes_to_levels(
            box_lists, self.min_level, self.max_level, self.canonical_box_size, self.canonical_level
        )

        num_boxes = len(pooler_fmt_boxes)
        num_channels = x[0].shape[1]
        output_size = self.output_size[0]

        dtype, device = x[0].dtype, x[0].device
        output = torch.zeros(
            (num_boxes, num_channels, output_size, output_size), dtype=dtype, device=device
        )

        # Contains index of which batch/image each ROI belongs to
        batch_index = torch.zeros(num_boxes, dtype=dtype, device=device)

        for level, (x_level, pooler) in enumerate(zip(x, self.level_poolers)):
            inds = torch.nonzero(level_assignments == level).squeeze(1)
            pooler_fmt_boxes_level = pooler_fmt_boxes[inds]
            output[inds] = pooler(x_level, pooler_fmt_boxes_level)
            batch_index[inds] = pooler_fmt_boxes_level[:, 0]

        return output, batch_index

