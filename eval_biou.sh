cd ./biou
python ./tools/coco_instance_evaluation.py \
      --gt-json-file /home/wqr/datasets/datasets/coco/annotations/instances_val2017.json \
      --dt-json-file /home/wqr/detection/DCT-Mask/projects/dct_seperate_1_test/output/42x42_resize_112x112/inference/coco_instances_results.json \
      --iou-type boundary