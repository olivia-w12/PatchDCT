CONFIG=patchdct_r50_city_1x
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python train_net.py --config-file configs/PatchDCT/${CONFIG}.yaml \
                    --num-gpus 4 \
                    --dist-url "tcp://127.0.0.1:6020" \
                    MODEL.WEIGHTS output/patchdct_r50_1x/model_final.pth