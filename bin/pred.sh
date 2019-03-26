echo "来预测图片的字块识别了"

python main/pred.py \
    --debug_mode=True \
    --save=True \
    --evaluate_split=False \
    --test_home=data/test \
    --pred_home=data/pred \
    --file=0.png \
    --gpu=0 \
    --model=model