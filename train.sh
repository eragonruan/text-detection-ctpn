CUDA_VISIBLE_DEVICES=1 \
python main/train.py \
	--pretrained_model_path=data/vgg_16.ckpt \
	--max_steps=4000 \
    >> ./logs_mlt//ctpn.log 2>&1