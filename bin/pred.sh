if [ "$1" = "help" ]; then
    echo "pred.sh 使用说明：
    --evaluate      是否进行评价（你可以光预测，也可以一边预测一边评价）
    --split         是否对小框做出评价，和画到图像上
    --test_home     被预测的图片目录
    --pred_home     预测后的结果的输出目录
    --file          为了支持单独文件，如果为空，就预测test_home中的所有文件
    --draw          是否把gt和预测画到图片上保存下来，保存目录也是pred_home
    --save          是否保存输出结果（大框、小框信息都要保存），保存到pred_home目录里面去
    --model         model的存放目录，会自动加载最新的那个模型 "
    exit
fi

echo "开始检测图片的字块区域....."

python main/pred.py \
    --debug_mode=True \
    --save=True \
    --file=0.png \
    --evaluate=True \
    --split=True \
    --test_home=data/test \
    --pred_home=data/pred \
    --draw=True \
    --gpu=0 \
    --model=model