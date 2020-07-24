#!/bin/sh
# 这个脚本用来把图片切割成CTPN训练所需要的bbox，即小框（CTPN算法是预测所有的16像素的小框的）

BASEDIR=$(dirname "$0")
. $BASEDIR/env.sh
echo "运行环境设置：$BASEDIR/env.sh"

if [ "$1" = "" ]; then
    echo "Usage: imsplit.sh <type:train|test|validate|xxxx>"
    exit
fi

# 默认处理
images_dir=data/$1/images
labels_split_dir=data/$1/labels.split
labels_dir=data/$1/labels
raw_images_dir=data/$1/raw.images
raw_labels_dir=data/$1/raw.labels

if [ "$1" = "train" ]; then
    images_dir=$train_images_dir
    labels_split_dir=$train_labels_split_dir
    labels_dir=$train_labels_dir
    raw_images_dir=$train_raw_images_dir
    raw_labels_dir=$train_raw_labels_dir
fi

if [ "$1" = "validate" ]; then
    images_dir=$validate_images_dir
    labels_split_dir=$validate_labels_split_dir
    labels_dir=$validate_labels_dir
    raw_images_dir=$validate_raw_images_dir
    raw_labels_dir=$validate_raw_labels_dir
fi

python -m utils.prepare.split_label \
    --images_dir=$images_dir \
    --labels_split_dir=$labels_split_dir \
    --labels_dir=$labels_dir \
    --raw_images_dir=$raw_images_dir \
    --raw_labels_dir=$raw_labels_dir