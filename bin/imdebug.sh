#!/bin/sh

if [ "$2" = "" ]; then
    echo "用来把GT和bbox框画到图像上，或者，用来验证是不是符合anchor大于400的阈值"
    echo "Usage: imdraw.sh <action:draw|anchor> <image_dir> <label_dir> <split_dir>"
    echo "例如：bin/imdebug.sh anchor data/train/images data/train/labels data/train/labels.split>log.txt 2>&1 &"
    exit
fi

python -m utils.prepare.image_debug \
    --action=$1 \
    --image_dir=$2 \
    --label_dir=$3 \
    --label_split_dir=$4
