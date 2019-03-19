if [ "$2" = "" ]; then
    echo "Usage: imdraw.sh <type:train|test|validate> <dir:data>"
    exit
fi

python data_generator/draw.py --type=$1 --dir=$2