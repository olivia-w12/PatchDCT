
def add_dctmask_config(cfg):
    """
    Add config for DCT-Mask.
    """

    # For MaskRCNNDCTHead
    cfg.MODEL.ROI_MASK_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    cfg.MODEL.ROI_MASK_HEAD.DCT_VECTOR_DIM = 300
    cfg.MODEL.ROI_MASK_HEAD.DCT_VECTOR_DIM_FIRST = 1 #add
    cfg.MODEL.ROI_MASK_HEAD.DCT_VECTOR_DIM_CUT = 300 #add
    cfg.MODEL.ROI_MASK_HEAD.MASK_SIZE = 128
    cfg.MODEL.ROI_MASK_HEAD.DCT_LOSS_TYPE = "l1"
    cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_PARA = 1.0
    cfg.MODEL.ROI_MASK_HEAD.MASK_LOSS_BALANCE_PARA = 10
