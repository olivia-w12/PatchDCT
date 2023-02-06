CONFIG=patchdct_r50_1x
CUDA_VISIBLE_DEVICES=2 \
python train_net.py --config-file configs/PatchDCT/${CONFIG}.yaml \
                    --eval-only \
                    --num-gpus 1 --resume \
                    --dist-url "tcp://127.0.0.1:6017" \
                    SOLVER.IMS_PER_BATCH 1 \
                    MODEL.WEIGHTS \
                    output/${CONFIG}/model_final.pth

