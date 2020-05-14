# `text-detection-ctpn`

Scene text detection based on CTPN (connectionist text proposal network). It is implemented in Tensorflow. The origin paper can be found [here](https://arxiv.org/abs/1609.03605). Also, the origin repo in Caffe can be found in [here](https://github.com/tianzhi0549/CTPN). For more details about the paper and code, see this [blog](http://slade-ruan.me/2017/10/22/text-detection-ctpn/). If you have any questions, check the issues first; if the problem persists, open a new issue.

***

**NOTICE: Thanks to [banjin-xjy](https://github.com/banjin-xjy), banjin and I have reonstructed this repo. The old repo was written based on Faster-RCNN, and remains tons of useless code and dependencies, make it hard to understand and maintain. Hence we reonstruct this repo. The old code is saved in [branch master](https://github.com/eragonruan/text-detection-ctpn/tree/master)**

***

# Roadmap

- [x] reonstruct the repo
- [x] cython nms and bbox utils
- [x] loss function as referred in paper
- [x] oriented text connector
- [x] BLSTM

***

# Setup

## General steps

### Windows

1.  Install Python3 64-bit
    -   from the [official site](https://www.python.org/downloads/), ***or***
    -   with Anaconda, from the [official site](https://www.anaconda.com/products/individual#Downloads), ***or***
    -   with Anaconda, [via Chocolatey](https://chocolatey.org/packages/anaconda3): `choco install anaconda3`
2.  Open a terminal where you have access to Python
    -   this might not be `cmd`, depending if you chose to add `python` to `%PATH%` during installation
    -   to make sure the installation succeeded, run: `python --version`
3.  From the terminal, go into the root directory of the project: `cd <ROOT>`
4.  To install dependencies:
    -   automatically, run the shell script: `.\infra\prepare.bat`, ***or***
    -   manually, follow the instructions in the next section

### Linux

1.  Install Python3 64-bit: `sudo apt install python3 python3-pip`
2.  To make sure installation the succeeded, run `python --version`
3.  From the terminal, go into the root directory of the project: `cd <ROOT>`
4.  To install dependencies:
    -   automatically, run the shell script: `./infra/prepare.sh`, ***or***
    -   manually, follow the instructions in the next section

## Manually installing the dependencies

If for any reason you would like not to run the `prepare.*` scripts, or if you encounter any issues, these are the steps you need to take; use the scripts as your guide:

1.  Install Cython
2.  Go into `<ROOT>/utils/bbox/` and build `nms` and `bbox`, which need Cython
3.  Install the latest Tensorflow 1.x
4.  Download the trained model from [Google Drive](https://drive.google.com/file/d/1HcZuB_MHqsKhKEKpfF1pEU85CYy4OlWO/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1BNHt_9fiqRPGmEXPaxaFXw)
5.  Unzip the archive into `<ROOT>`; more precisely, put `checkpoints_mlt/` in `text-detection-ctpn/`

***

# Running the demo

1.  Put your input images in `data/demo`
2.  Run the demo, for example from the root dir:
    ```shell
    python ./main/demo.py
    ```
3.  Find the results in `data/res`

***

# Training

## Prepare the data

### Using the existing dataset

1.  Download the pre-trained model of VGG net and put it in `data/vgg_16.ckpt`. you can download it from [tensorflow/models](https://github.com/tensorflow/models/tree/1af55e018eebce03fb61bba9959a04672536107d/research/slim)
2.  Download the dataset we prepared from [Google Drive](https://drive.google.com/file/d/1npxA_pcEvIa4c42rho1HgnfJ7tamThSy/view?usp=sharing) or [Baidu Yun](https://pan.baidu.com/s/1nbbCZwlHdgAI20_P9uw9LQ)
3.  Put the downloaded data in `data/dataset/mlt`, then start the training.

### Using your own dataset

1.  Modify the DATA_FOLDER and OUTPUT in `utils/prepare/split_label.py` according to your dataset
2.  Run `split_label.py` in the root:
    ```shell
    python ./utils/prepare/split_label.py
    ```
3.  It will generate the prepared data in `data/dataset/`
4.  The input file format demo of `split_label.py` can be found in [gt_img_859.txt](https://github.com/eragonruan/text-detection-ctpn/blob/banjin-dev/data/readme/gt_img_859.txt), and the output file of `split_label.py` is [img_859.txt](https://github.com/eragonruan/text-detection-ctpn/blob/banjin-dev/data/readme/img_859.txt). A demo image of the prepared data is shown below.
<img src="/data/readme/demo_split.png" width=640 height=480 />

***

## Training

1.  Simply run:
    ```shell
    python ./main/train.py
    ```
2.  The model provided in `checkpoints_mlt` is trained on GTX1070 for 50k iters. It takes about 0.25s per iter. So it will take about 3.5 hours to finish 50k iterations.

***

# Some results

**NOTICE:** all the photos used below are collected from the internet. If it affects you, please contact me to delete them.
<img src="/data/res/006.jpg" width=320 height=480 /><img src="/data/res/008.jpg" width=320 height=480 />
<img src="/data/res/009.jpg" width=320 height=480 /><img src="/data/res/010.png" width=320 height=320 />

***

## Oriented text connector

-   Oriented text connector has been implemented; it's working, but still needs further improvements
-   Left figure is the result for DETECT_MODE H, right figure is for DETECT_MODE O
<img src="/data/res/007.jpg" width=320 height=240 /><img src="/data/res_oriented/007.jpg" width=320 height=240 />
<img src="/data/res/008.jpg" width=320 height=480 /><img src="/data/res_oriented/008.jpg" width=320 height=480 />

***
