import torch
from .utils import masks2patch
from detectron2.layers import cat

class GT_infomation:
    def __init__(self,mask_size_assemble, mask_size, patch_size, scale, dct_encoding, patch_dct_encoding):
        self.mask_size_assemble = mask_size_assemble
        self.mask_size = mask_size
        self.patch_size = patch_size
        self.scale = scale
        self.dct_encoding = dct_encoding
        self.patch_dct_encoding =  patch_dct_encoding
    
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
            gt_masks_per_image = masks2patch(gt_masks_per_image,self.scale,self.patch_size,self.mask_size_assemble)
            gt_masks.append(gt_masks_per_image)

            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0
        gt_masks = cat(gt_masks, dim=0)
        gt_masks = self.patch_dct_encoding.encode(gt_masks)  # [N, dct_v_dim]
        gt_masks = gt_masks.to(dtype=torch.float32) #[N_instance,pdct_vector_dim]
        gt_classes = cat(gt_classes, dim=0) #[N_instanc]
        gt_masks_coarse = cat(gt_masks_coarse,dim=0)
        gt_masks_coarse = self.dct_encoding.encode(gt_masks_coarse).to(dtype=torch.float32)
        gt_masks, gt_bfg = self.get_gt_bfg(gt_masks)
        return gt_masks,gt_classes,gt_masks_coarse,gt_bfg


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

            gt_masks_per_image = masks2patch(gt_masks_per_image,self.scale,self.patch_size,self.mask_size_assemble)
            gt_masks.append((gt_masks_per_image))

        if len(gt_masks) == 0:
            return pred_mask_logits.sum() * 0

        gt_masks = cat(gt_masks, dim=0)
        gt_masks = self.patch_dct_encoding.encode(gt_masks)
        gt_masks = gt_masks.to(dtype=torch.float32)
        gt_masks, gt_bfg = self.get_gt_bfg(gt_masks)
        return gt_masks,gt_bfg

    def get_gt_bfg(self, gt_masks):
        gt_bfg = gt_masks[:, 0].clone()
        gt_bfg[(gt_bfg > 0) & (gt_bfg < self.patch_size)] = 1.
        gt_bfg[gt_bfg == self.patch_size] = 2
        gt_bfg = gt_bfg.to(dtype=torch.int64)
        gt_masks = gt_masks[gt_bfg == 1, :]
        return gt_masks, gt_bfg
    
    def get_gt_classes(self, instances):
        gt_classes = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
            gt_classes.append(gt_classes_per_image)
        gt_classes = cat(gt_classes, dim=0)  # [N_instance]
        return gt_classes
