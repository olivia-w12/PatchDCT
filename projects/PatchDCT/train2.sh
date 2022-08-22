CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python train_net.py --config-file configs/DCT/refine_dct_vectors.yaml \
                    --num-gpus 8 \
                    --dist-url "tcp://127.0.0.1:6020"