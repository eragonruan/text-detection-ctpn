# 这个脚本用来生成样本，目前已经废弃，因为生成的样本效果不好，后来索性用真实样本了

if [ "$3" = "" ]; then
    echo "Usage: imagen.sh <type:train|test|validate> <dir:data> <num:100>"
    exit
fi

python data_generator/generator.py --type=$1 --dir=$2 --num=$3