from typing import List

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm
from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import BaseMaskRCNNHead
from detectron2.structures import Instances
from .mask_encoding import DctMaskEncoding


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNPatchDCTHead_NSTAGE(BaseMaskRCNNHead):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, dct_vector_dim, mask_size,
                 fine_features_resolution,mask_size_assemble,patch_size,
                 patch_dct_vector_dim,mask_loss_para,dct_loss_type,
                 num_stage,mask_loss_para_each_stage,
                 conv_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of classes. 1 if using class agnostic prediction.For COCO,num_classes=80
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
            dct_vector_dim: dct vector dim in DCT_MASK(default=300)
            mask_size: resolution of mask to be refined(default=112)
            fine_features_resolution: feature map in PatchDCT(default 42x42)
            mask_size_assemble: mask size in PatchDCT(default=112)
            patch_size: patch size(default=8)
            patch_dct_vector_dim: DCT vector dim for each patch

        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"

        self.patch_dct_vector_dim = patch_dct_vector_dim
        self.mask_size_assemble = mask_size_assemble
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.hidden_features = 1024
        self.dct_vector_dim = dct_vector_dim
        self.mask_size = mask_size
        self.dct_loss_type = dct_loss_type
        self.mask_loss_para = mask_loss_para
        self.scale = self.mask_size // self.patch_size
        self.ratio = fine_features_resolution // self.scale
        self.num_stage = num_stage-1
        self.loss_para = mask_loss_para_each_stage

        self.dct_encoding_coarse = DctMaskEncoding(vec_dim=self.dct_vector_dim, mask_size=self.mask_size)
        self.dct_encoding = DctMaskEncoding(vec_dim=self.patch_dct_vector_dim, mask_size=self.patch_size)

        self.conv_norm_relus = []

        cur_channels = input_shape.channels
        for k, conv_dim in enumerate(conv_dims[:-1]):
            conv = Conv2d(
                cur_channels,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("mask_fcn{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            cur_channels = conv_dim



        self.predictor = nn.Sequential(
            nn.Linear(14**2*conv_dim,self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features,self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features,self.dct_vector_dim)
        )
        self.reshape = Conv2d(
            1,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            activation=F.relu
        )
        self.fusion = nn.Sequential(
            Conv2d(cur_channels ,
                   conv_dim,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   bias=not conv_norm,
                   norm=get_norm(conv_norm, conv_dim),
                   activation=F.relu),
            Conv2d(cur_channels,
                   conv_dim,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   bias=not conv_norm,
                   norm=get_norm(conv_norm, conv_dim),
                   activation=F.relu)
        )


        self.downsample= nn.Sequential(
            Conv2d(
                cur_channels,
                self.hidden_features,
                kernel_size=self.ratio,
                stride=self.ratio,
                padding=0,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu, ),
            Conv2d(self.hidden_features,
                   self.hidden_features,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   bias=not conv_norm,
                   norm=get_norm(conv_norm, conv_dim),
                   activation=F.relu),
        )

        self.predictor1 = Conv2d(self.hidden_features,
                                               self.patch_dct_vector_dim*self.num_classes,
                                               kernel_size=1,
                                               stride=1,
                                               padding=0,
                                               )
        self.predictor_bfg = Conv2d(self.hidden_features,
                                    3*self.num_classes,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    )

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)


    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        conv_dim = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        num_conv = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        ret.update(
            conv_dims=[conv_dim] * (num_conv + 1),  # +1 for ConvTranspose
            conv_norm=cfg.MODEL.ROI_MASK_HEAD.NORM,
            input_shape=input_shape,
            dct_vector_dim=cfg.MODEL.ROI_MASK_HEAD.DCT_VECTOR_DIM,
            mask_loss_para=cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_PARA,
            mask_size=cfg.MODEL.ROI_MASK_HEAD.MASK_SIZE,
            dct_loss_type=cfg.MODEL.ROI_MASK_HEAD.DCT_LOSS_TYPE,
            fine_features_resolution = cfg.MODEL.ROI_MASK_HEAD.FINE_FEATURES_RESOLUTION,
            mask_size_assemble = cfg.MODEL.ROI_MASK_HEAD.MASK_SIZE_ASSEMBLE,
            patch_size = cfg.MODEL.ROI_MASK_HEAD.PATCH_SIZE,
            patch_dct_vector_dim = cfg.MODEL.ROI_MASK_HEAD.PATCH_DCT_VECTOR_DIM,
            num_stage = cfg.MODEL.ROI_MASK_HEAD.NUM_STAGE,
            mask_loss_para_each_stage = cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_PARA_EACH_STAGE
        )

        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def layers(self, x,fine_mask_features0,instances):
        """

        Args:
            x: feature map used in DCT-Mask
            fine_mask_features: feature map used in PatchDCT

        Returns:
            x (Tensor): [B, D]. D is dct-dim. [B, D]. DCT_Vector in DCT-Mask.
            bfg: A tensor of shape [B,num_class,3,scale,scale].
                A NxN masks is divided into scale x scale patches.
                bfg demonstrates results of three-class classifier in PatchDCT
                0 for foreground,1 for mixed,2 for background
            detail : A tensor of shape:[B,num_class,patch_dct_vector_dim,scale,scale].
                    DCT vector for each patch (only calculate loss for mixed patch)
        """
        for layer in self.conv_norm_relus:
            x = layer(x)
        #------------------------------DCT-Mask--------------------------
        x = self.predictor(x.flatten(start_dim=1))
        if not self.training:
            num_masks = x.shape[0]
            if num_masks==0:
                return x, {},{}
        #reverse transform to obtain high-resolution masks
        masks = self.dct_encoding_coarse.decode(x).real.reshape(-1,1,self.mask_size,self.mask_size)
        # ------------------------------DCT-Mask--------------------------

        # ------------------------------PatchDCT--------------------------
        bfg,detail = self.patchdct(fine_mask_features0,masks)

        # ------------------------------PatchDCT--------------------------

        # ------------------------------PatchDCT--------------------------
        if self.training:
            classes = self.get_gt_classes(instances)
        else:
            classes = instances[0].pred_classes
        num_instance = classes.size()[0]
        indice = torch.arange(num_instance)

        bfg_dict = {}
        detail_dict = {}

        bfg = bfg[indice, classes].permute(0, 2, 3, 1).reshape(-1, 3)
        detail = detail[indice, classes].permute(0, 2, 3, 1).reshape(-1, self.patch_dct_vector_dim)

        bfg_dict[0] = bfg
        detail_dict[0] = detail

        for i in range(1,self.num_stage):
            masks = self.stage_patch2mask(bfg,detail)
            bfg,detail = self.patchdct(fine_mask_features0,masks)
            bfg = bfg[indice, classes].permute(0, 2, 3, 1).reshape(-1, 3)
            detail = detail[indice, classes].permute(0, 2, 3, 1).reshape(-1, self.patch_dct_vector_dim)
            bfg_dict[i] = bfg
            detail_dict[i] = detail

        return x,bfg_dict,detail_dict

    def patchdct(self,fine_mask_features0,masks):
        masks = F.interpolate(masks, size=(self.scale * self.ratio, self.scale * self.ratio))
        masks = self.reshape(masks)
        fine_mask_features = masks + fine_mask_features0
        fine_mask_features = self.fusion(fine_mask_features)
        fine_mask_features = self.downsample(fine_mask_features)
        detail = self.predictor1(fine_mask_features)
        bfg = self.predictor_bfg(fine_mask_features)
        bfg = bfg.reshape(-1, self.num_classes, 3, self.scale, self.scale)
        detail = detail.reshape(-1, self.num_classes, self.patch_dct_vector_dim, self.scale, self.scale)
        return bfg,detail

    def stage_patch2mask(self,bfg,detail):
        device = bfg.device
        index = torch.argmax(bfg, dim=1)
        bg = torch.zeros_like(detail, device=device)
        bg[index == 1] = 1
        fg = torch.zeros_like(detail, device=device)
        fg[index == 2, 0] = self.patch_size
        masks = detail * bg + fg
        masks = self.dct_encoding.decode(masks).real
        masks = self.patch2masks(masks,self.scale,self.patch_size,self.mask_size_assemble)
        return masks[:,None,:,:]

    def forward(self, x, fine_mask_features,instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        x,bfg_dict,detail_dict = self.layers(x,fine_mask_features,instances)
        if self.training:
            return {"loss_mask": self.mask_rcnn_dct_loss(x,bfg_dict,detail_dict,instances, self.vis_period)}
        else:

            pred_instances = self.mask_rcnn_dct_inference(x,bfg_dict,detail_dict,instances)
            return pred_instances

    def mask_rcnn_dct_loss(self, pred_mask_logits,bfg_dict,detail_dict,instances, vis_period=0):
        """
        Compute the mask prediction loss defined in the Mask R-CNN paper.

        Args:
            pred_mask_logits (Tensor): [B, D]. D is dct-dim. [B, D]. DCT_Vector in DCT-Mask.
            bfg: A tensor of shape [B,num_class,3,scale,scale].
                A NxN masks is divided into scale x scale patches.
                bfg demonstrates results of three-class classifier in PatchDCT
                0 for foreground,1 for mixed,2 for background
            detail : A tensor of shape:[B,num_class,patch_dct_vector_dim,scale,scale].
                    DCT vector for each patch (only calculate loss for mixed patch)
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            vis_period (int): the period (in steps) to dump visualization.

        Returns:
            mask_loss (Tensor): A scalar tensor containing the loss.
        """
        gt_masks,gt_classes,gt_masks_coarse= self.get_gt_mask(instances,pred_mask_logits)
        gt_bfg = gt_masks[:, 0].clone()
        gt_bfg[(gt_bfg > 0) & (gt_bfg < self.patch_size)] = 1.
        gt_bfg[gt_bfg == self.patch_size] = 2
        gt_bfg = gt_bfg.to(dtype = torch.int64)

        gt_masks = gt_masks[gt_bfg==1,:]

        if self.dct_loss_type == "l1":
            mask_loss = F.l1_loss(pred_mask_logits, gt_masks_coarse)
            mask_loss = mask_loss*self.loss_para[0]
            for i in range(self.num_stage):
                bfg = bfg_dict[i]
                detail = detail_dict[i]
                detail = detail[gt_bfg==1]
                mask_loss += self.loss_para[i+1]*(F.cross_entropy(bfg,gt_bfg) + F.l1_loss(detail,gt_masks))

            mask_loss = self.mask_loss_para * mask_loss
            
        elif self.dct_loss_type == "sl1":
            num_instance = gt_masks.size()[0]
            mask_loss = F.smooth_l1_loss(pred_mask_logits, gt_masks, reduction="none")
            mask_loss = self.mask_loss_para * mask_loss / num_instance
            mask_loss = torch.sum(mask_loss)
        elif self.dct_loss_type == "l2":
            num_instance = gt_masks.size()[0]
            mask_loss = F.mse_loss(pred_mask_logits, gt_masks, reduction="none")
            mask_loss = self.mask_loss_para * mask_loss / num_instance
            mask_loss = torch.sum(mask_loss)
        else:
            raise ValueError("Loss Type Only Support : l1, l2; yours: {}".format(self.dct_loss_type))

        return mask_loss

    def mask_rcnn_dct_inference(self,pred_mask_logits,bfg_dict,detail_dict,pred_instances):
        """
        Convert pred_mask_logits to estimated foreground probability masks while also
        extracting only the masks for the predicted classes in pred_instances. For each
        predicted box, the mask of the same class is attached to the instance by adding a
        new "pred_masks" field to pred_instances.

        Args:
            pred_mask_logits (Tensor): A tensor of shape (B, C, Hmask, Wmask) or (B, 1, Hmask, Wmask)
                for class-specific or class-agnostic, where B is the total number of predicted masks
                in all images, C is the number of foreground classes, and Hmask, Wmask are the height
                and width of the mask predictions. The values are logits.
            bfg: A tensor of shape [B,num_class,3,scale,scale].
                A NxN masks is divided into scale x scale patches.
                bfg demonstrates results of three-class classifier in PatchDCT
                0 for foreground,1 for mixed,2 for background
            detail : A tensor of shape:[B,num_class,patch_dct_vector_dim,scale,scale].
                    DCT vector for each patch (only calculate loss for mixed patch)
            pred_instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. Each Instances must have field "pred_classes".

        Returns:
            None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
                Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
                masks the resolution predicted by the network; post-processing steps, such as resizing
                the predicted masks to the original image resolution and/or binarizing them, is left
                to the caller.
        """

        num_patch = pred_mask_logits.shape[0]
        device = pred_mask_logits.device
        if num_patch == 0:
            pred_instances[0].pred_masks = torch.empty([0, 1, self.mask_size, self.mask_size]).to(device)
            return pred_instances
        else:

            # pred_classes = pred_instances[0].pred_classes
            # num_masks = pred_classes.shape[0]
            # indices = torch.arange(num_masks)
            # bfg = bfg[indices,pred_classes].permute(0,2,3,1).reshape(-1,3)
            # detail = detail[indices,pred_classes].permute(0,2,3,1).reshape(-1,self.patch_dct_vector_dim)

            with torch.no_grad():
                n = self.num_stage - 1
                bfg = bfg_dict[n]
                detail = detail_dict[n]
                bfg = F.softmax(bfg,dim=1)
                threshold = 0.36
                bfg[bfg[:,0]>threshold,0] = bfg[bfg[:,0]>threshold,0]+1
                bfg[bfg[:,2]>threshold,2] = bfg[bfg[:,2]>threshold,2]+1
                index = torch.argmax(bfg,dim=1)

                # ---------------------------gt-------------------------------
                # gt_masks = self.get_gt_mask_inference(pred_instances, pred_mask_logits)
                # gt_bfg = gt_masks[:, 0].clone()
                # gt_bfg[(gt_bfg > 0) & (gt_bfg < self.patch_size)] = 1.
                # gt_bfg[gt_bfg == self.patch_size] = 2
                # gt_bfg = gt_bfg.to(dtype=torch.int64)
                # ---------------------------gt-------------------------------

                detail[index == 0, ::] = 0
                detail[index == 2, ::] = 0
                detail[index == 2, 0] = self.patch_size

                pred_mask_rc = self.dct_encoding.decode(detail)
                # assemble patches to obtain an entire mask
                pred_mask_rc = self.patch2masks(pred_mask_rc,self.scale,self.patch_size,self.mask_size_assemble)


            pred_mask_rc = pred_mask_rc[:, None, :, :]
            pred_instances[0].pred_masks = pred_mask_rc
            return pred_instances

    def get_gt_classes(self,instances):
        gt_classes = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        gt_classes = cat(gt_classes, dim=0) #[N_instanc]
        return gt_classes

    def get_gt_mask(self,instances,pred_mask_logits):
        gt_masks = []
        gt_classes = []
        gt_masks_coarse = []
        for instances_per_image in instances:

            if len(instances_per_image) == 0:
                continue

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, self.mask_size)
            gt_masks_coarse.append(gt_masks_per_image)

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, self.mask_size_assemble)
            # divided masks into scalexscale patch,patch size=8
            gt_masks_per_image = self.masks2patch(gt_masks_per_image,self.scale,self.patch_size,self.mask_size_assemble)

            gt_masks.append(gt_masks_per_image)

            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0
        gt_masks = cat(gt_masks, dim=0)
        gt_masks = self.dct_encoding.encode(gt_masks)  # [N, dct_v_dim]
        gt_masks = gt_masks.to(dtype=torch.float32) #[N_instance,pdct_vector_dim]
        gt_classes = cat(gt_classes, dim=0) #[N_instanc]
        gt_masks_coarse = cat(gt_masks_coarse,dim=0)
        gt_masks_coarse = self.dct_encoding_coarse.encode(gt_masks_coarse).to(dtype=torch.float32)
        return gt_masks,gt_classes,gt_masks_coarse


    def get_gt_mask_inference(self,instances,pred_mask_logits):
        gt_masks = []

        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if instances_per_image.has("gt_masks"):
                gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.pred_boxes.tensor, self.mask_size_assemble)
            else:
                #print("gt_mask is empty")
                shape = instances_per_image.pred_boxes.tensor.shape[0]
                device = instances_per_image.pred_boxes.tensor.device
                gt_masks_per_image = torch.zeros((shape,self.mask_size_assemble,self.mask_size_assemble),dtype=torch.bool).to(device)

            gt_masks_per_image = self.masks2patch(gt_masks_per_image,self.scale,self.patch_size,self.mask_size_assemble)
            gt_masks_vector = self.dct_encoding.encode(gt_masks_per_image)
            gt_masks.append((gt_masks_vector))

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)
        gt_masks = gt_masks.to(dtype=torch.float32)
        return gt_masks

    def patch2masks(self,patch,scale,patch_size,mask_size):
        """
        assemble patches to obtain an entire mask
        Args:
            mask_rc: A tensor of shape [B*num_patch,patch_size,patch_size]
            scale: A NxN mask size is divided into scale x scale patches
            patch_size: size of each patch
            mask_size: size of masks generated by PatchDCT

        Returns:
            A tensor of shape [B,mask_size,mask_size].The masks obtain assemble by patches
        """
        patch = patch.reshape(-1, scale, scale, patch_size, patch_size)
        patch = patch.permute(0, 1, 2, 4, 3)
        patch = patch.reshape(-1, scale, mask_size, patch_size)
        patch = patch.permute(0, 1, 3, 2)
        mask = patch.reshape(-1, mask_size, mask_size)
        return mask

    def masks2patch(self,masks_per_image,scale,patch_size,mask_size):
        """

        Args:
            masks_per_image: A tensor of shape [B,mask_size,mask_size]
            scale: A NxN mask size is divided into scale x scale patches
            patch_size: size of each patch
            mask_size: size of masks generated by PatchDCT

        Returns:
            patches_per_image: A tensor of shape [B*num_patch,patch_size,patch_size]. The patches obtained by masks

        """
        masks_per_image = masks_per_image.reshape(-1, scale, patch_size,mask_size)
        masks_per_image = masks_per_image.permute(0, 1, 3, 2)
        masks_per_image = masks_per_image.reshape(-1, scale, scale, patch_size, patch_size)
        masks_per_image = masks_per_image.permute(0, 1, 2, 4, 3)
        patches_per_image = masks_per_image.reshape(-1,patch_size, patch_size)
        return patches_per_image