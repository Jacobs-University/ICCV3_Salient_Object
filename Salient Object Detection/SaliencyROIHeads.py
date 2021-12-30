#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from typing import Dict

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels

from SaliencyFast_RCNN import SaliencyFastRCNNOutputLayers, SaliencyFastRCNNOutputs

from SaliencyBoxHead import build_saliency_box_head
from SaliencyMaskHead import build_mask_head, mask_rcnn_inference, mask_rcnn_loss
from SaliencyROIPooler import ROIPooler

import torch
import gc
gc.collect()


def select_foreground_proposals(proposals, bg_label):
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Instances)
    assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)
    return fg_proposals, fg_selection_masks


class CustomROIHeads(torch.nn.Module):
#ROIHeads perform all per-region computation in an R-CNN.


    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super(CustomROIHeads, self).__init__()

        # fmt: off
        self.batch_size_per_image     = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE    # 512
        self.positive_sample_fraction = cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION       # 0.25
        self.test_score_thresh        = 0.5                                         # 0.05 original for object detection
        self.test_nms_thresh          = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST         # 0.5
        self.test_detections_per_img  = cfg.TEST.DETECTIONS_PER_IMAGE               # 100
        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES             # ["res4"]
        self.proposal_append_gt       = cfg.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT      # True
        self.feature_strides          = {k: v.stride for k, v in input_shape.items()}
        self.feature_channels         = {k: v.channels for k, v in input_shape.items()}
        self.cls_agnostic_bbox_reg    = cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG    # False
        self.smooth_l1_beta           = cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA
        self.num_classes              = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        # fmt: on

        # Matcher to assign box proposals to gt boxes
        self.proposal_matcher = Matcher(
            cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS,
            cfg.MODEL.ROI_HEADS.IOU_LABELS,
            allow_low_quality_matches=False,
        )

        # Box2BoxTransform for bounding box regression
        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS)

    def _sample_proposals(self, matched_idxs, matched_labels, gt_classes):
#Based on the matching between N proposals and M groundtruth,sample the proposals and set their classification labels.

        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
            gt_classes[matched_labels == 0] = self.num_classes
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, self.num_classes
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_proposals(self, proposals, targets):
#It performs box matching between `proposals` and `targets`, and assigns training labels to the proposals.

        gt_boxes = [x.gt_boxes for x in targets]
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            # Set target attributes of the sampled proposals:
            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            # We index all the attributes of targets that start with "gt_"
            # and have not been added to proposals yet (="gt_classes").
            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]

                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        proposals_per_image.set(trg_name, trg_value[sampled_targets])
            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples))
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples))

        return proposals_with_gt

    def forward(self, images, features, proposals, scene_context_feat, targets=None):
        raise NotImplementedError()


class SaliencyROIHeads(CustomROIHeads):
    def __init__(self, cfg, input_shape):
        super(SaliencyROIHeads, self).__init__(cfg, input_shape)

        self._init_saliency_box_head(cfg)
        self._init_mask_head(cfg)

    def _init_saliency_box_head(self, cfg):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION    # 7
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)  # (0.25, 0.125, 0.0625, 0.03125)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO   # 0
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE    # "ROIAlignV2"
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [self.feature_channels[f] for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        self.box_head = build_saliency_box_head(cfg)
        self.box_predictor = SaliencyFastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

    def _init_mask_head(self, cfg):
        # fmt: off
        self.mask_on = cfg.MODEL.MASK_ON
        if not self.mask_on:
            return
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION   # 14
        pooler_scales = tuple(1.0 / self.feature_strides[k] for k in self.in_features)  # (0.25, 0.125, 0.0625, 0.03125)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO  # 0
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE   # "ROIAlignV2"
        # fmt: on

        in_channels = [self.feature_channels[f] for f in self.in_features][0]

        self.mask_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.mask_head = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )

    def forward(self, images, features, proposals, scene_context_feat, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        features_list = [features[f] for f in self.in_features]

        if self.training:
            losses = self._forward_box(features_list, proposals, scene_context_feat)
            # During training the proposals used by the box head are
            # used by the mask, keypoint (and densepose) heads.
            losses.update(self._forward_mask(features_list, proposals))

            return proposals, losses
        else:
            pred_instances = self._forward_box(features_list, proposals, scene_context_feat)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)

            return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")
        features = [features[f] for f in self.in_features]

        instances = self._forward_mask(features, instances)

        return instances

    def _forward_box(self, features, proposals, scene_context_feat):

        box_features, box_batch_index = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features, box_batch_index, scene_context_feat)

        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)

        del box_features

        outputs = SaliencyFastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances

    def _forward_mask(self, features, instances):
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features, mask_batch_index = self.mask_pooler(features, proposal_boxes)
            mask_logits = self.mask_head(mask_features)

            return {"loss_mask": mask_rcnn_loss(mask_logits, proposals)}
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features, mask_batch_index = self.mask_pooler(features, pred_boxes)
            mask_logits = self.mask_head(mask_features)
            mask_rcnn_inference(mask_logits, instances)
            return instances


def build_saliency_roi_heads(cfg, input_shape):
    return SaliencyROIHeads(cfg, input_shape)

