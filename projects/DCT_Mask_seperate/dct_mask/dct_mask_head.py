

import fvcore.nn.weight_init as weight_init
from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.config import configurable
from detectron2.modeling.roi_heads.mask_head import BaseMaskRCNNHead
from detectron2.layers import Conv2d, ShapeSpec, cat, get_norm,ConvTranspose2d
from detectron2.structures import Instances

from .mask_encoding import DctMaskEncoding


@ROI_MASK_HEAD_REGISTRY.register()
class MaskRCNNDCTHead(BaseMaskRCNNHead):
    """
    A mask head with several conv layers, plus an upsample layer (with `ConvTranspose2d`).
    Predictions are made with a final 1x1 conv layer.
    """

    @configurable
    def __init__(self, input_shape: ShapeSpec, *, num_classes, dct_vector_dim, mask_size,
                 mask_loss_para,
                 dct_loss_type,
                 conv_dims, conv_norm="", **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            num_classes (int): the number of classes. 1 if using class agnostic prediction.
            conv_dims (list[int]): a list of N>0 integers representing the output dimensions
                of N-1 conv layers and the last upsample layer.
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(**kwargs)
        assert len(conv_dims) >= 1, "conv_dims have to be non-empty!"
        self.dct_vector_dim = dct_vector_dim
        self.mask_size = mask_size

        self.scale = 16
        self.ratio = 2
        self.hidden_features = 1024
        self.dct_size = self.mask_size//self.scale
        self.dct_loss_type = dct_loss_type
        self.mask_loss_para = mask_loss_para


        
        self.dct_encoding = DctMaskEncoding(vec_dim=self.dct_vector_dim, mask_size=self.dct_size)

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

        self.fusion1 = Conv2d(cur_channels * 2,
                              conv_dim,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=not conv_norm,
                              norm=get_norm(conv_norm, conv_dim),
                              activation=F.relu)

        self.mixture1 = Conv2d(
            cur_channels,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu, )
        self.resize = Conv2d(
            cur_channels,
            self.hidden_features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, self.hidden_features),
            activation=F.relu, )
        self.predictor_fg = nn.Sequential(
            nn.Linear(self.hidden_features,self.hidden_features),
            nn.ReLU(),
            nn.Linear(self.hidden_features,3)
        )
        """
        self.predictor_fg = nn.Sequential(
            Conv2d(
                cur_channels,
                self.hidden_features,
                kernel_size=self.ratio,
                stride=self.ratio,
                padding=0,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu, ),
            ConvTranspose2d(self.hidden_features,3,kernel_size=2,stride=2)
        )
        """
        self.fusion_20 = Conv2d(cur_channels*2+3,
                                conv_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=not conv_norm,
                                norm=get_norm(conv_norm, conv_dim),
                                activation=F.relu)
        self.fusion_21= Conv2d(conv_dim,
                              1,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=not conv_norm,
                              norm=get_norm(conv_norm, conv_dim),
                              activation=F.relu)
        self.fusion_22 = Conv2d(conv_dim,
                                3,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=not conv_norm,
                                norm=get_norm(conv_norm, conv_dim),
                                activation=F.relu)
        self.fusion_23 = Conv2d(conv_dim,
                                5,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=not conv_norm,
                                norm=get_norm(conv_norm, conv_dim),
                                activation=F.relu)
        self.fusion_24 = Conv2d(cur_channels*2+9,
                                conv_dim,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=not conv_norm,
                                norm=get_norm(conv_norm, conv_dim),
                                activation=F.relu)
        self.mixture2 = Conv2d(
            cur_channels,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu, )

        self.downsample= Conv2d(
            cur_channels,
            self.hidden_features,
            kernel_size=self.ratio,
            stride=self.ratio,
            padding=0,
            bias=not conv_norm,
            norm=get_norm(conv_norm, conv_dim),
            activation=F.relu, )
        self.predictor1 = nn.Sequential(nn.Linear(self.hidden_features,self.hidden_features),
                                        nn.ReLU(),
                                        nn.Linear(self.hidden_features,self.dct_vector_dim))

        #self.predictor_first_dim = nn.Linear(1024,1)
        #self.predictor = nn.Linear(1024,self.dct_vector_dim-1)

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        """
        for m in [self.predictor_first_dim]:
            nn.init.normal_(m.weight, std=0.1)
            nn.init.constant_(m.bias, 0)

        nn.init.normal_(self.predictor.weight,std=0.01)
        nn.init.constant_(self.predictor.bias,0)
        """

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

        )

        if cfg.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK:
            ret["num_classes"] = 1
        else:
            ret["num_classes"] = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        return ret

    def layers(self, x,fine_mask_features):
        for layer in self.conv_norm_relus:
            x = layer(x)
        x = self.fusion1(torch.cat((x, F.interpolate(fine_mask_features,scale_factor=1/self.ratio)), dim=1))
        x = self.mixture1(x)
        fg = self.resize(x)
        fg = self.predictor_fg(fg.permute(0,2,3,1)).permute(0,3,1,2)
        #fg = self.predictor_fg(x)

        x = F.interpolate(x,scale_factor=self.ratio)
        x = self.fusion_20(torch.cat((x, F.interpolate(fg,scale_factor=self.ratio),fine_mask_features), dim=1))
        x1 = self.fusion_21(x)
        x2 = self.fusion_22(x)
        x3 = self.fusion_23(x)
        x = self.fusion_24(torch.cat((x,fine_mask_features,x1,x2,x3),dim=1))
        x = self.mixture2(x)

        fg = fg.permute(0, 2, 3, 1).reshape(-1, 3)
        x= self.downsample(x) #[N,C,16,16]
        vector = x.permute(0,2,3,1) #[N,16,16,C]
        vector = self.predictor1(vector)
        vector = vector.reshape(-1,self.dct_vector_dim)

        vector_head = vector[:,:1]
        vector = vector[:,1:]

        return vector, vector_head,fg

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
        x,head,fg = self.layers(x,fine_mask_features,)
        if self.training:
            return {"loss_mask": self.mask_rcnn_dct_loss(x,head,fg,instances, self.vis_period)}
        else:
            pred_instances = self.mask_rcnn_dct_inference(x,head,fg,instances)
            return pred_instances

    def mask_rcnn_dct_loss(self, pred_mask_logits,head, fg,instances, vis_period=0):
        """
        Compute the mask prediction loss defined in the Mask R-CNN paper.

        Args:
            pred_mask_logits (Tensor): [B, D]. D is dct-dim. [B, D]. DCT_Vector.
            
            instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. These instances are in 1:1
                correspondence with the pred_mask_logits. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.
            vis_period (int): the period (in steps) to dump visualization.

        Returns:
            mask_loss (Tensor): A scalar tensor containing the loss.
        """
        gt_masks = self.get_gt_mask(instances,pred_mask_logits)
        gt_fg = gt_masks[:,0].clone()
        gt_fg [(gt_fg>0)&(gt_fg<8)] = 1
        gt_fg [gt_fg==8] = 2
        gt_fg = gt_fg.to(dtype = torch.int64)
        gt_masks = gt_masks[gt_fg==1,:]
        pred_mask_logits = pred_mask_logits[gt_fg==1,:]
        head = head[gt_fg==1,:]


        if self.dct_loss_type == "l1":
            #num_instance = gt_masks.size()[0]
            mask_loss_1 = F.l1_loss(pred_mask_logits, gt_masks[:, 1:])
            mask_loss_2 = F.l1_loss(head, gt_masks[:, :1] / self.dct_size)
            mask_loss_3 = F.cross_entropy(fg,gt_fg)
            mask_loss = mask_loss_1 + mask_loss_2 + mask_loss_3
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

    def mask_rcnn_dct_inference(self,pred_mask_logits,head,fg,pred_instances):
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
            pred_instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. Each Instances must have field "pred_classes".

        Returns:
            None. pred_instances will contain an extra "pred_masks" field storing a mask of size (Hmask,
                Wmask) for predicted class. Note that the masks are returned as a soft (non-quantized)
                masks the resolution predicted by the network; post-processing steps, such as resizing
                the predicted masks to the original image resolution and/or binarizing them, is left
                to the caller.
        """

        num_masks = pred_mask_logits.shape[0]
        device = pred_mask_logits.device

        if num_masks == 0:
            pred_instances[0].pred_masks = torch.empty([0, 1, self.mask_size, self.mask_size]).to(device)
            return pred_instances
        else:

            with torch.no_grad():
                pred_mask_logits = torch.cat((head*self.dct_size,pred_mask_logits),dim=1)
                gt_masks = self.get_gt_mask_inference(pred_instances, pred_mask_logits)


                gt_fg = gt_masks[:, 0].clone()
                gt_fg[(gt_fg > 0) & (gt_fg < 8)] = 1
                gt_fg[gt_fg == 8] = 2
                gt_fg = gt_fg.to(dtype=torch.int64)


                index = fg.argmax(dim=1)
                #loss_1 = F.l1_loss(pred_mask_logits,gt_masks,reduction="none")
                #loss_2 = F.cross_entropy(fg,gt_fg,reduction="none")
                #print(loss_1[gt_fg==1,:].mean(dim=0))
                #print(loss_2.mean())
                #index = gt_fg
                pred_mask_logits[index==1,:] = gt_masks[index==1,:]
                pred_mask_logits[index == 0,::] = 0
                pred_mask_logits[index==2,::] = 0
                pred_mask_logits[index==2,0] = self.dct_size

                pred_mask_rc = self.dct_encoding.decode(pred_mask_logits)

                pred_mask_rc = pred_mask_rc.reshape(-1, self.scale, self.scale, self.dct_size, self.dct_size)
                pred_mask_rc = pred_mask_rc.permute(0, 1, 2, 4, 3)
                pred_mask_rc = pred_mask_rc.reshape(-1, self.scale, self.mask_size, self.dct_size)
                pred_mask_rc = pred_mask_rc.permute(0, 1, 3, 2)

                pred_mask_rc = pred_mask_rc.reshape(-1, self.mask_size, self.mask_size)


            pred_mask_rc = pred_mask_rc[:, None, :, :]
            pred_instances[0].pred_masks = pred_mask_rc
            return pred_instances

    def get_gt_mask(self,instances,pred_mask_logits):
        gt_masks = []
        for instances_per_image in instances:

            if len(instances_per_image) == 0:
                continue

            gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                instances_per_image.proposal_boxes.tensor, self.mask_size)

            gt_masks_per_image = gt_masks_per_image.reshape(-1, self.scale, self.dct_size, self.mask_size)
            gt_masks_per_image = gt_masks_per_image.permute(0, 1, 3, 2)
            gt_masks_per_image = gt_masks_per_image.reshape(-1, self.scale, self.scale, self.dct_size, self.dct_size)
            gt_masks_per_image = gt_masks_per_image.permute(0, 1, 2, 4, 3)

            gt_masks_per_image = gt_masks_per_image.reshape(-1, self.dct_size, self.dct_size)
            gt_masks_vector = self.dct_encoding.encode(gt_masks_per_image)  # [N, dct_v_dim]
            gt_masks.append(gt_masks_vector)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0
        gt_masks = cat(gt_masks, dim=0)
        gt_masks = gt_masks.to(dtype=torch.float32)
        return gt_masks


    def get_gt_mask_inference(self,instances,pred_mask_logits):
        gt_masks = []

        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            if instances_per_image.has("gt_masks"):
                gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
                    instances_per_image.pred_boxes.tensor, self.mask_size)
            else:
                #print("gt_mask is empty")
                shape = instances_per_image.pred_boxes.tensor.shape[0]
                device = instances_per_image.pred_boxes.tensor.device
                gt_masks_per_image = torch.zeros((shape,self.mask_size,self.mask_size),dtype=torch.bool).to(device)


            #gt_masks_vector = self.dct_encoding.encode(gt_masks_per_image)  # [N, dct_v_dim]
            #gt_masks.append(gt_masks_vector)
            gt_masks_per_image = gt_masks_per_image.reshape(-1, self.scale, self.dct_size, self.mask_size)
            gt_masks_per_image = gt_masks_per_image.permute(0, 1, 3, 2)
            gt_masks_per_image = gt_masks_per_image.reshape(-1, self.scale, self.scale, self.dct_size, self.dct_size)
            gt_masks_per_image = gt_masks_per_image.permute(0, 1, 2, 4, 3)

            gt_masks_per_image = gt_masks_per_image.reshape(-1, self.dct_size, self.dct_size)
            gt_masks_vector = self.dct_encoding.encode(gt_masks_per_image)
            gt_masks.append((gt_masks_vector))

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)

        gt_masks = gt_masks.to(dtype=torch.float32)
        return gt_masks
