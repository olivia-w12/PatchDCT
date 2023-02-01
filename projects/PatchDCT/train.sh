CONFIG=patchdct_r50_city_1x_fine
CUDA_VISIBLE_DEVICES=1,2,3,4 \
python train_net.py --config-file configs/PatchDCT/${CONFIG}.yaml \
                    --num-gpus 4 \
                    --dist-url "tcp://127.0.0.1:6020"
