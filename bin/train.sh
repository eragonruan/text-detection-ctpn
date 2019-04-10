# 默认CTPN用GPU1，CRNN用GPU0

if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep vgg|awk '{print $2}'|xargs kill -9
    exit
fi

if [ "$1" = "console" ]; then
    echo "调试模式"
    CUDA_VISIBLE_DEVICES=1 \
    python main/train.py \
        --pretrained_model_path=data/vgg_16.ckpt \
        --max_steps=1 \
        --decay_steps=1 \
        --evaluate_steps=1 \
        --learning_rate=0.01 \
        --save_checkpoint_steps=2000 \
        --decay_rate=0.1 \
        --lambda1=400 \
        --debug_mode=True \
        --logs_path=logs \
        --moving_average_decay=0.997 \
        --restore=False
    exit
fi

if [ "$1" = "gpu0" ]; then
    echo "生产模式:GPU0"
    CUDA_VISIBLE_DEVICES=0 \
    python main/train.py \
        --pretrained_model_path=data/vgg_16.ckpt \
        --max_steps=100000 \
        --decay_steps=10000 \
        --evaluate_steps=5000 \
        --learning_rate=0.0001 \
        --save_checkpoint_steps=5000 \
        --decay_rate=0.5 \
        --lambda1=400 \
        --gpu=0 \
        --debug_mode=False \
        --logs_path=logs \
        --moving_average_decay=0.997 \
        --restore=False \
        >> ./logs/ctpn_gpu0.log 2>&1
    exit
fi

if [ "$1" = "gpu1" ]; then
    echo "生产模式:GPU1"
    CUDA_VISIBLE_DEVICES=1 \
    python main/train.py \
        --pretrained_model_path=data/vgg_16.ckpt \
        --max_steps=40000 \
        --decay_steps=8000 \
        --evaluate_steps=5000 \
        --learning_rate=0.0001 \
        --save_checkpoint_steps=5000 \
        --lambda1=1 \
        --decay_rate=0.3 \
        --debug_mode=False \
        --logs_path=logs \
        --moving_average_decay=0.997 \
        --restore=False \
        >> ./logs/ctpn_gpu1.log 2>&1
    exit
fi