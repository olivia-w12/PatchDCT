CUDA_VISIBLE_DEVICES=1,2,3,4 \
python train_net.py --config-file configs/DCT/test.yaml\
                    --eval-only \
                    --num-gpus 4 --resume \
                    --dist-url "tcp://127.0.0.1:6017"
