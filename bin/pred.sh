# 这个脚本用于偶尔测试一下探测效果用的，会生成对应的预测结果，并画在一张输出图上

if [ "$1" = "help" ]; then
    echo "pred.sh 使用说明：
    --evaluate      是否进行评价（你可以光预测，也可以一边预测一边评价）
    --split         是否对小框做出评价，和画到图像上
    --image_dir     被预测的图片目录
    --pred_dir     预测后的结果的输出目录
    --file          为了支持单独文件，如果为空，就预测test_home中的所有文件
    --draw          是否把gt和预测画到图片上保存下来，保存目录也是pred_dir
    --save          是否保存输出结果（大框、小框信息都要保存），保存到pred_dir目录里面去
    --ctpn_model_dir         model的存放目录，会自动加载最新的那个模型 "
    exit
fi

echo "开始检测图片的字块区域....."

python main/pred.py \
    --debug=True \
    --save=True \
    --evaluate=True \
    --split=False \
    --test_dir=data/test \
    --image_name=0.png \
    --pred_dir=data/pred \
    --draw=True \
    --ctpn_model_dir=model \
    --ctpn_model_file=ctpn_50000.ckpt