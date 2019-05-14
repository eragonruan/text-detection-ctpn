Date=$(date +%Y%m%d%H%M)

. bin/env.sh

if [ "$1" = "stop" ]; then
    echo "停止训练"
    ps aux|grep python|grep ctpn|awk '{print $2}'|xargs kill -9
    exit
fi

if [ "$1" = "console" ]; then
    echo "调试模式:只训练一次"
    python -m main.train \
        --name=ctpn \
        --pretrained_model_path=data/vgg_16.ckpt \
        --max_steps=2 \
        --decay_steps=1 \
        --evaluate_steps=1 \
        --validate_batch=1 \
        --learning_rate=0.01 \
        --decay_rate=0.1 \
        --lambda1=1000 \
        --resize=True \
        --num_readers=1 \
        --debug=True \
        --train_images_dir=$train_images_dir \
        --train_labels_split_dir=$train_labels_split_dir \
        --train_labels_dir=$train_labels_dir \
        --validate_images_dir=$validate_images_dir \
        --validate_labels_split_dir=$validate_labels_split_dir \
        --validate_labels_dir=$validate_labels_dir
    exit
fi

echo "生产模式,使用GPU#$1"
nohup python -m main.train \
    --name=ctpn \
    --pretrained_model_path=data/vgg_16.ckpt \
    --max_steps=100000 \
    --decay_steps=15000 \
    --evaluate_steps=500 \
    --train_images_dir=$train_images_dir \
    --train_labels_split_dir=$train_labels_split_dir \
    --train_labels_dir=$train_labels_dir \
    --validate_images_dir=$validate_images_dir \
    --validate_labels_split_dir=$validate_labels_split_dir \
    --validate_labels_dir=$validate_labels_dir \
    --validate_batch=5 \
    --learning_rate=0.0001 \
    --decay_rate=0.3 \
    --lambda1=10000 \
    --gpu=$1 \
    --debug=False \
    --logs_path=logs/tboard \
    --moving_average_decay=0.997 \
    --restore=False \
    --early_stop=20 \
    --max_lr_decay=3 \
    >> ./logs/ctpn_gpu$1_$Date.log 2>&1 &
