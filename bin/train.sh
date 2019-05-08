Date=$(date +%Y%m%d%H%M)

if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep vgg|awk '{print $2}'|xargs kill -9
    exit
fi

if [ "$1" = "console" ]; then
    echo "调试模式:只训练一次"
    python -m main.train \
        --pretrained_model_path=data/vgg_16.ckpt \
        --max_steps=2 \
        --decay_steps=1 \
        --evaluate_steps=1 \
        --validate_dir=data/validate \
        --validate_batch=1 \
        --train_dir=data/train \
        --learning_rate=0.01 \
        --save_checkpoint_steps=2000 \
        --decay_rate=0.1 \
        --lambda1=1000 \
        --gpu=1\
        --debug=True \
        --logs_path=logs \
        --moving_average_decay=0.997 \
        --restore=False \
        --early_stop=5
    exit
fi

echo "生产模式,使用GPU#$1"
nohup python -m main.train \
    --pretrained_model_path=data/vgg_16.ckpt \
    --max_steps=100000 \
    --decay_steps=10000 \
    --evaluate_steps=1000 \
    --validate_dir=data/validate \
    --validate_batch=10 \
    --train_dir=data/train \
    --learning_rate=0.0001 \
    --save_checkpoint_steps=5000 \
    --decay_rate=0.5 \
    --lambda1=1000 \
    --gpu=$1 \
    --debug=False \
    --logs_path=logs \
    --moving_average_decay=0.997 \
    --restore=False \
    --early_stop=5 \
    --max_lr_decay=3 \
    >> ./logs/ctpn_gpu$1_$Date.log 2>&1 &
