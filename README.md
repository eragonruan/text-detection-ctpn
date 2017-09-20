# text-detection-ctpn
text detection mainly based on ctpn (connectionist text proposal network) model in tensorflow, id card detect,[arxiv1609.03605](https://arxiv.org/abs/1609.03605),[caffe github](https://github.com/tianzhi0549/CTPN)
***
# introduce
- this repo is mainly based on faster rcnn framework, so there remains tons of useless code. I'm still working on it.
- the model is able to work. the training part is already include in this repo, but I haven't  organize the code of how to prepare the training data. So I won't explain about how to train the model for now. If you are intersted in the training part, feel free to contact me, I can provide you with the prepared traiing data. Or you can prepare the training data as referred in ctpn paper.
- Also, this model is mainly based on CTPN, but without side-refinement part.
***
# demo
- put your images in data/demo, the results will be saved in data/results
- run python ctpn/demo.py in the root directory.
***
# some results
`NOTICE:` all the photos used below are collected from the internet. If it affects you, please contact me to delete them.
<img src="/data/results/001.jpg" width=320 height=240 /><img src="/data/results/002.jpg" width=320 height=240 />
<img src="/data/results/006.jpg" width=320 height=240 /><img src="/data/results/008.jpg" width=320 height=240 />
***
