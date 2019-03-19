if [ "$3" = "" ]; then
    echo "Usage: imagen.sh <type:train|test|validate> <dir:data> <num:100>"
    exit
fi

python data_generator/generator.py --type=$1 --dir=$2 --num=$3