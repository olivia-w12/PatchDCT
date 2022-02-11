# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import  ROIPooler
from detectron2.modeling.roi_heads.mask_head import build_mask_head

from detectron2.modeling.roi_heads import StandardROIHeads,select_foreground_proposals

from .utils import add_ground_truth_to_pred_boxes


@ROI_HEADS_REGISTRY.register()
class ROIHeads_MASK_DIFFER(StandardROIHeads):
    """
    Use given feature to generate mask layer
    """

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features = cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES
        # in_features       = ("p2", )
        # print('forcing use P2 as mask InFeatures')
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        ret["mask_head"] = build_mask_head(
            cfg, ShapeSpec(channels=in_channels, width=pooler_resolution, height=pooler_resolution)
        )
        return ret
    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
            del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.match_gt_to_pred_boxes(targets,pred_instances)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


    def match_gt_to_pred_boxes(self,targets,pred_instances):
        with torch.no_grad():
            pred_instances_with_gt = []
            for pred_instances_per_image, targets_per_image in zip(pred_instances, targets):
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, pred_instances_per_image.pred_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)

                #match_gt
                has_gt = len(targets_per_image) > 0
                # Get the corresponding GT for each proposal
                if has_gt:
                    gt_classes = targets_per_image.gt_classes[matched_idxs]
                    # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                    gt_classes[matched_labels == 0] = self.num_classes
                    # Label ignore proposals (-1 label)
                    gt_classes[matched_labels == -1] = -1
                else:
                    gt_classes = torch.zeros_like(matched_idxs) + self.num_classes


                pred_instances_per_image.gt_classes = gt_classes

                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                if has_gt:
                    for (trg_name, trg_value) in targets_per_image.get_fields().items():
                        if trg_name.startswith("gt_") and not pred_instances_per_image.has(trg_name):
                            trg_value = trg_value[matched_idxs]
                            pred_instances_per_image.set(trg_name, trg_value)
                else:
                    gt_boxes = Boxes(
                        targets_per_image.gt_boxes.tensor.new_zeros((len(matched_idxs), 4))
                    )
                    pred_instances_per_image.gt_boxes = gt_boxes

                pred_instances_with_gt.append(pred_instances_per_image)


        return pred_instances_with_gt


    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask_inference(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_mask_inference(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the predicted boxes.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.mask_in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, _ = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            return self.mask_head(mask_features, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            return self.mask_head(mask_features, instances)