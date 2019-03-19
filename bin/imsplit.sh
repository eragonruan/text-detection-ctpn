if [ "$2" = "" ]; then
    echo "Usage: imsplit.sh <type:train|test|validate> <dir:data>"
    exit
fi

python utils/prepare/split_label.py --type=$1 --dir=$2