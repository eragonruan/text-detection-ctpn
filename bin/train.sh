CUDA_VISIBLE_DEVICES=1 \
python main/train.py \
	--pretrained_model_path=data/vgg_16.ckpt \
	--max_steps=40000 \
	--decay_steps=2000 \
	--evaluate_steps=10 \
	--learning_rate=0.001 \
	--save_checkpoint_steps=2000 \
	--decay_rate=0.5 \
	--debug_mode=False \
	--logs_path=logs \
	--moving_average_decay=0.997 \
	--restore=False \
    >> ./logs/ctpn.log 2>&1


# 0.5 ^ 10 = 0.001
# 0.5 ^ 20 = 10e-6