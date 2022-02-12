
def add_dctmask_config(cfg):
    """
    Add config
    """

    cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION2 = 28
    cfg.MODEL.ROI_MASK_HEAD.GT_MASKS_RESOLUTION = 128


