if [ "$2" = "" ]; then
    echo "Usage: imsplit.sh <type:train|test|validate> <dir:data>"
    exit
fi

python -m utils.prepare.split_label --type=$1 --dir=$2