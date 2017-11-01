# text-detection-ctpn

text detection mainly based on ctpn (connectionist text proposal network). It is implemented in tensorflow. I use id card detect as an example to demonstrate the results, but it should be noticing that this model can be used in almost every horizontal scene text detection task. The origin paper can be found [here](https://arxiv.org/abs/1609.03605). Also, the origin repo in caffe can be found in [here](https://github.com/tianzhi0549/CTPN). This repo is mainly based on faster rcnn framework, so there remains tons of useless code. I'm still working on it. For more detail about the paper and code, see this [blog](http://slade-ruan.me/2017/10/22/text-detection-ctpn/)
***

# prepare

First, download the pre-trained model of VGG net and put it in data/pretrain/VGG_imagenet.npy. you can download it from [google drive](https://drive.google.com/open?id=0B_WmJoEtfQhDRl82b1dJTjB2ZGc).  
Second, prepare the training data as referred in paper, or you can download the data I prepared in [here](https://drive.google.com/open?id=0B_WmJoEtfQhDRl82b1dJTjB2ZGc). Modify the path and gt_path in prepare_training_data/split_label.py according to your dataset. And run
```shell
cd prepare_training_data
python split_label.py
```
it will generate the prepared data in current folder, and then run
```shell
python ToVoc.py
```
to convert the prepared training data into voc format. It will generate a folder named TEXTVOC. move this folder to data/ and then run
```shell
cd ../data
ln -s TEXTVOC VOCdevkit2007
```
***

# demo
- to run the demo, you have to build nms implement in cython or cuda first
- you can change USE_GPU_NMS in ctpn/text.yml to decide whether to use gpu nms or cython nms
```shell
cd lib/utils
chmod +x make.sh
./make.sh
```
- then put your images in data/demo, the results will be saved in data/results, and run demo in the root
```shell
python ./ctpn/demo.py
```
***

# train
Simplely run
```shell
python ./ctpn/train_net.py
```
you can modify some hyper parameters in ctpn/text.yml, or just used the parameters I set.
***


# roadmap
- [x] cython nms
- [x] cuda nms
- [x] python2/python3 compatblity
- [x] tensorflow1.3
- [x] delete useless code
- [x] loss function as referred in paper
- [ ] side refinement
- [ ] model optimization

------

# some results
`NOTICE:` all the photos used below are collected from the internet. If it affects you, please contact me to delete them.
<img src="/data/results/001.jpg" width=320 height=240 /><img src="/data/results/004.jpg" width=320 height=240 />
<img src="/data/results/006.jpg" width=320 height=480 /><img src="/data/results/009.jpg" width=320 height=480 />
<img src="/data/results/008.jpg" width=320 height=480 /><img src="/data/results/010.jpg" width=320 height=320 />
***
