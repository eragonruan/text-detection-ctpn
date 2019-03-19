CUDA_VISIBLE_DEVICES=1 \
python main/train.py \
	--pretrained_model_path=data/vgg_16.ckpt \
	--learning_rate=0.1 \
	--max_steps=16000 \
	--decay_steps=4000 \
	--decay_rate=0.1\
    >> ./logs_mlt/ctpn.log 2>&1