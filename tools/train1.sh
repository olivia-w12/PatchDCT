CUDA_VISIBLE_DEVICES=1,2,3,4 \
python train_net.py --config-file configs/Cityscapes/mask_rcnn_R_50_FPN.yaml \
                    --num-gpus 4 \
                    --dist-url "tcp://127.0.0.1:6017"