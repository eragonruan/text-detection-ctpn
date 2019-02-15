# text-detection-ctpn

Scene text detection based on ctpn (connectionist text proposal network). It is implemented in tensorflow. The origin paper can be found [here](https://arxiv.org/abs/1609.03605). Also, the origin repo in caffe can be found in [here](https://github.com/tianzhi0549/CTPN). For more detail about the paper and code, see this [blog](http://slade-ruan.me/2017/10/22/text-detection-ctpn/). If you got any questions, check the issue first, if the problem persists, open a new issue.
***
**NOTICE: Thanks to [banjin-xjy](https://github.com/banjin-xjy), banjin and I have reonstructed this repo. The old repo was written based on Faster-RCNN, and remains tons of useless code and dependencies, make it hard to understand and maintain. Hence we reonstruct this repo. The old code is saved in [branch master](https://github.com/eragonruan/text-detection-ctpn/tree/master)**
***
# roadmap
- [x] reonstruct the repo
- [x] cython nms and bbox utils
- [x] loss function as referred in paper
- [x] oriented text connector
- [x] BLSTM
***
# setup
nms and bbox utils are written in cython, hence you have to build the library first.
```shell
cd utils/bbox
chmod +x make.sh
./make.sh
```
It will generate a nms.so and a bbox.so in current folder.
***
# demo
- follow setup to build the library 
- download the ckpt file from [googl drive](https://drive.google.com/file/d/1HcZuB_MHqsKhKEKpfF1pEU85CYy4OlWO/view?usp=sharing) or [baidu yun](https://pan.baidu.com/s/1BNHt_9fiqRPGmEXPaxaFXw)
- put checkpoints_mlt/ in text-detection-ctpn/
- put your images in data/demo, the results will be saved in data/res, and run demo in the root 
```shell
python ./main/demo.py
```
***
# training
## prepare data
- First, download the pre-trained model of VGG net and put it in data/vgg_16.ckpt. you can download it from [tensorflow/models](https://github.com/tensorflow/models/tree/1af55e018eebce03fb61bba9959a04672536107d/research/slim)
- Second, download the dataset we prepared from [google drive](https://drive.google.com/file/d/1npxA_pcEvIa4c42rho1HgnfJ7tamThSy/view?usp=sharing) or [baidu yun](https://pan.baidu.com/s/1nbbCZwlHdgAI20_P9uw9LQ). put the downloaded data in data/dataset/mlt, then start the training.
- Also, you can prepare your own dataset according to the following steps. 
- Modify the DATA_FOLDER and OUTPUT in utils/prepare/split_label.py according to your dataset. And run split_label.py in the root
```shell
python ./utils/prepare/split_label.py
```
- it will generate the prepared data in data/dataset/
- The input file format demo of split_label.py can be found in [gt_img_859.txt](https://github.com/eragonruan/text-detection-ctpn/blob/banjin-dev/data/readme/gt_img_859.txt). And the output file of split_label.py is [img_859.txt](https://github.com/eragonruan/text-detection-ctpn/blob/banjin-dev/data/readme/img_859.txt). A demo image of the prepared data is shown below.
<img src="/data/readme/demo_split.png" width=640 height=480 />

***
## train 
Simplely run
```shell
python ./main/train.py
```
- The model provided in checkpoints_mlt is trained on GTX1070 for 50k iters. It takes about 0.25s per iter. So it will takes about 3.5 hours to finished 50k iterations.
***
# some results
`NOTICE:` all the photos used below are collected from the internet. If it affects you, please contact me to delete them.
<img src="/data/res/006.jpg" width=320 height=480 /><img src="/data/res/008.jpg" width=320 height=480 />
<img src="/data/res/009.jpg" width=320 height=480 /><img src="/data/res/010.png" width=320 height=320 />
***
## oriented text connector
- oriented text connector has been implemented, i's working, but still need futher improvement.
- left figure is the result for DETECT_MODE H, right figure for DETECT_MODE O
<img src="/data/res/007.jpg" width=320 height=240 /><img src="/data/res_oriented/007.jpg" width=320 height=240 />
<img src="/data/res/008.jpg" width=320 height=480 /><img src="/data/res_oriented/008.jpg" width=320 height=480 />
***
