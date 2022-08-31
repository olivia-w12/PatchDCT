python ./biou/tools/coco_instance_evaluation.py \
      --gt-json-file ./datasets/coco/annotations/instances_val2017.json \
      --dt-json-file ./output/patch_size_4/inference/coco_instances_results.json \
      --iou-type boundary