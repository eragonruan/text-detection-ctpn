#!/bin/sh

# Set root directory
self_path=$(readlink -f "$0")
root_dir=$(dirname "$self_path")/..

# Install Cython
python3 --version || exit 1
pip3 install --upgrade Cython || exit 1

# Build Cython dependencies
cd $root_dir/utils/bbox
python3 setup.py build
cd build/lib*
cp *.so ../..
cd ../..
rm -rf build

# Install training backend
pip3 install tensorflow==1.14

# Download trained model
cd $root_dir/infra
if [ ! -f checkpoints_mlt.zip ]; then
    python3 download_gdrive_file.py 1HcZuB_MHqsKhKEKpfF1pEU85CYy4OlWO checkpoints_mlt.zip
fi

# Extract trained model
echo "Extracting, this may take some time"
unzip -q -o -d $root_dir checkpoints_mlt.zip
