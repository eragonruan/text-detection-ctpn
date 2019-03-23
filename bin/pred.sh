echo "来预测图片的字块识别了"

python main/pred.py \
    --debug_mode=True \
    --home=data/test \
    --file=0.png \
    --gpu=0 \
    --model=checkpoints_mlt
