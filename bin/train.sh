if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep vgg|awk '{print $2}'|xargs kill -9
    exit
fi

CUDA_VISIBLE_DEVICES=1 \
python main/train.py \
	--pretrained_model_path=data/vgg_16.ckpt \
	--max_steps=100000 \
	--decay_steps=2000 \
	--evaluate_steps=1000 \
	--learning_rate=0.1 \
	--save_checkpoint_steps=2000 \
	--decay_rate=0.5 \
	--debug_mode=False \
	--logs_path=logs \
	--moving_average_decay=0.997 \
	--restore=False \
    >> ./logs/ctpn.log 2>&1


# 0.5 ^ 10 = 0.001
# 0.5 ^ 20 = 10e-6