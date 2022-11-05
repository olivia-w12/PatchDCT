CUDA_VISIBLE_DEVICES=1,2,3,4 \
python train_net.py --config-file configs/DCT/R50_1x_dct_300_l1_0_007_4conv.yaml \
                    --num-gpus 4 \
                    --dist-url "tcp://127.0.0.1:6018" \
                    OUTPUT_DIR "output/test1/"
