# text-detection-ctpn

Scene text detection based on ctpn (connectionist text proposal network). It is implemented in tensorflow. The origin paper can be found [here](https://arxiv.org/abs/1609.03605). Also, the origin repo in caffe can be found in [here](https://github.com/tianzhi0549/CTPN). For more detail about the paper and code, see this [blog](http://slade-ruan.me/2017/10/22/text-detection-ctpn/). If you got any questions, check the issue first, if the problem persists, open a new issue.
***
**NOTICE: Thankd to [banjin-xjy](https://github.com/banjin-xjy), banjin and I have reonstructed this repo. The old repo was written based on Faster-RCNN, and remains tons of useless code and dependencies, make it hard to understand and maintain. Hencd we reonstruct this repo. The old code is saved in [branch master](https://github.com/eragonruan/text-detection-ctpn/tree/master)**
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
- download the ckpt file from [release](https://github.com/eragonruan/text-detection-ctpn/releases)
- put checkpoints_mlt/ in text-detection-ctpn/
- put your images in data/demo, the results will be saved in data/res, and run demo in the root 
```shell
python ./main/demo.py
```
***
# training
## prepare data
- First, download the pre-trained model of VGG net and put it in data/vgg_16.ckpt. you can download it from [google drive](https://drive.google.com/open?id=0B_WmJoEtfQhDRl82b1dJTjB2ZGc) or [baidu yun](https://pan.baidu.com/s/1kUNTl1l). 
- Second, prepare the training data according to the following steps. 
- Modify the DATA_FOLDER and OUTPUT in utils/prepare/split_label.py according to your dataset. And run split_label.py in the root
```shell
python ./utils/prepare/split_label.py
```
- it will generate the prepared data in data/dataset/
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
