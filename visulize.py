from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer,ColorMode
import cv2
import os
from tqdm import tqdm

config_file = "./configs/Cityscapes/mask_rcnn_R_50_FPN.yaml"
image_file = "./visualization/city_mask_rcnn/"
image_dir = "./visualization/image/"
image_paths = os.listdir(image_dir)
for i in tqdm(range(len(image_paths))):
    img_path = image_dir + image_paths[i]
    cfg = get_cfg()

    cfg.merge_from_file(config_file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    predictor  = DefaultPredictor(cfg)
    im = cv2.imread(img_path)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=None,
                   scale=1,
                   instance_mode=2   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imwrite(image_file+"{}.png".format(i),out.get_image()[:, :, ::-1])




