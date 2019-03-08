# 修改说明

- 增加了样本生成
- 增加了运行日志

生成样本用的背景、字体还有语料，太大，放到了百度云盘上：

[下载链接](https://pan.baidu.com/s/1CSJYUbsilRqm4zSfhxjZtQ) 提取码: 76j5 

# 样本生成

一共3个脚本：

__imgen.sh__

用来批量生成训练图片，使用了data_generator/background中的白纸背景，
然后从info.txt随机加载词语，随机选择字体，写到白纸上去，可以参数生成训练、验证和测试集合。
生成的目录为train/images和train/labels。

__imsplit.sh__

用来生成训练的标签，imgen.sh会产生图片和每一个句子对应的8个坐标，
但是，这不是这个程序要的样本，这个程序要的样本是更细的16个像素宽的小框，所以还需要再运行这个批处理，
从data/train/labels目录下读取一句话对应的8个坐标，然后生成一堆的更小的框的坐标。生成的目录为train/split。

注意，两个坐标不太一样，大框坐标是8个[x1,y1,x2,y2,x3,y3,x4,y4]，是为了支持任意四边形，
而小框都是矩形，所以只有4个坐标[x1,y1,x2,y2]，也就是左上和右下的坐标。

__imdraw.sh__

是读取刚才生成的大框和小框坐标，把这些框画到图上去，生成的目录为train/draw。

# 对程序的一些理解

程序还是有些复杂的，感谢 [@eragonruan/text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn),
真的是受益匪浅，认真阅读后，一些重要体会写在这里：

- 最复杂的函数就是anchor_target_layer，这个函数实际上是在干一件事，就是找到每一个anchor对应的分类和4个回归值
- 最开始加载的图片实际上只有一张，但是训练的时候是300，为何？是因为，程序从这一张图中，可能找到数万个anchor，可能有好几百个GT小框
所以，理论上对每一个anchor都可以做一个样本，但是实际上很多anchor和gt都是不想交的，所以最后只找出来300个，
其中正负样本的数量也是严格控制的，合在一起300个
- 为何是宽度是16个，是因为vgg之后，feature map恰好缩小了16倍，这样FM中的一个点对应回去就是16像素宽
- 整个过程是vgg16-conv-bilstm-fc/fc，最后的2个fc得到了bbox_pred[w,h,40]和cls_pred[w,h,20]，我奇怪为何bbox_pred最后的
维度是40，不是应该20么？也就是[d_y,d_h]么？这个还有些疑惑。至于为何是20，是因为需要分别预测10个anchor的前后景的概率。
- 总而言之，通过神经网络算出pred值，通过anchor_target_layer得到bbox/cls两者label值，然后做损失函数，酱紫

对于CTPN的整体理解，可以参考我的博客：["单据识别学习笔记"](http://www.piginzoo.com/machine-learning/2019/01/21/ocr)

# 运行日志

我打印了很多运行时候的日志，这样可以观察出整个一次的运行细节：

```
2019-03-08 13:13:54,840 : DEBUG : 开始运行sess.run了
libpng warning: iCCP: known incorrect sRGB profile
libpng warning: iCCP: known incorrect sRGB profile
2019-03-08 13:13:55.259518: I tensorflow/core/kernels/logging_ops.cc:79] 最开始输入[1 1023 808 3]
2019-03-08 13:14:07.276377: I tensorflow/core/kernels/logging_ops.cc:79] VGG的5-3卷基层输出[1 63 50 512]
2019-03-08 13:14:07.561730: I tensorflow/core/kernels/logging_ops.cc:79] LSTM输入[1 63 50 512]
2019-03-08 13:14:07.731334: I tensorflow/core/kernels/logging_ops.cc:79] LSTM输出[1 63 50 512]
2019-03-08 13:14:07.731378: I tensorflow/core/kernels/logging_ops.cc:79] LSTM后的FC输入[1 63 50 512]
2019-03-08 13:14:07.731408: I tensorflow/core/kernels/logging_ops.cc:79] LSTM后的FC输入[1 63 50 512]
2019-03-08 13:14:07.737872: I tensorflow/core/kernels/logging_ops.cc:79] LSTM后的FC输出[1 63 50 40]
2019-03-08 13:14:07.737899: I tensorflow/core/kernels/logging_ops.cc:79] bbox_pred[1 63 50 40]
2019-03-08 13:14:07.737936: I tensorflow/core/kernels/logging_ops.cc:79] Loss输入：bbox_pred[1 63 50 40]
2019-03-08 13:14:07.738514: I tensorflow/core/kernels/logging_ops.cc:79] LSTM后的FC输出[1 63 50 20]
2019-03-08 13:14:07.738536: I tensorflow/core/kernels/logging_ops.cc:79] cls_pred[1 63 50 20]
2019-03-08 13:14:07,739 : DEBUG : 开始调用anchor_target_layer，这个函数是来算anchor们和gt的差距
2019-03-08 13:14:07,740 : DEBUG : 传入的参数：
2019-03-08 13:14:07,740 : DEBUG : rpn_cls_score:<class 'numpy.ndarray'>
2019-03-08 13:14:07,741 : DEBUG : rpn_cls_score:(1, 63, 50, 20)
2019-03-08 13:14:07,741 : DEBUG : gt_boxes:<class 'numpy.ndarray'>
2019-03-08 13:14:07,742 : DEBUG : gt_boxes:(360, 5)
2019-03-08 13:14:07,742 : DEBUG : im_info:array([[1023.,  808.,    3.]], dtype=float32)
2019-03-08 13:14:07,743 : DEBUG : 得到了所有的anchor了：(10, 4)
2019-03-08 13:14:07,743 : DEBUG : 一共10个anchors
2019-03-08 13:14:07,743 : DEBUG : feature map H/W:(63,50)
2019-03-08 13:14:07,743 : DEBUG : shift_x (50,)
2019-03-08 13:14:07,744 : DEBUG : shift_y (63,)
2019-03-08 13:14:07,744 : DEBUG : 变换后shift_x (63, 50)
2019-03-08 13:14:07,744 : DEBUG : 变换后shift_y (63, 50)
2019-03-08 13:14:07,744 : DEBUG : shift_y,shift_x vstack后 (3150, 4)
2019-03-08 13:14:07,745 : DEBUG : 得到的all_anchors：(3150, 10, 4)
2019-03-08 13:14:07,745 : DEBUG : reshape后的all_anchors：(31500, 4)
2019-03-08 13:14:07,746 : DEBUG : 图像内的点的索引inds_inside：(29150,)
2019-03-08 13:14:07,746 : DEBUG : 图像内部的anchors：(29150, 4)
2019-03-08 13:14:07,746 : DEBUG : labels初始化：(29150,)
2019-03-08 13:14:07,777 : DEBUG : 经过bbox_overlaps处理后的overlops:(29150, 360)
2019-03-08 13:14:07,790 : DEBUG : 每行里面，最大的列号argmax_overlaps:(29150,)
2019-03-08 13:14:07,791 : DEBUG : 每行里面，最大的列号的值max_overlaps:(29150,)
2019-03-08 13:14:07,913 : DEBUG : 每列里面，最大的行号gt_argmax_overlaps:(360,)
2019-03-08 13:14:07,913 : DEBUG : 每列里面，最大的行号的值gt_max_overlaps:(360,)
2019-03-08 13:14:07,952 : DEBUG : np.where(overlaps == gt_max_overlaps):(array([ 1829,  1839,  1848,  1857,  1866,  1875,  1884,  1893,  1902,
        1911,  1920,  1964,  1974,  1983,  1992,  2001,  2010,  2019,
        2028,  2037,  2046,  2055,  2064,  2126,  2135,  2144,  2153,
        2162,  2738,  2747,  2756,  2765,  2774,  2783,  2792,  2801,
        2810,  2819,  2828,  2837,  2846,  2918,  2928,  2937,  2946,
        2955,  2964,  2973,  2982,  2991,  3000,  3009,  5202,  5212,
        5222,  5232,  5242,  5252,  5262,  5272,  5282,  5352,  5362,
        5372,  5382,  5392,  5402,  5412,  5422,  5432,  5442,  5452,
        5462,  6183,  6193,  6203,  6213,  6223,  6233,  6243,  6253,
        6263,  6273,  6283,  6293,  6303,  6351,  6363,  6373,  6383,
        6393,  6403,  6413,  6423,  6433,  6443,  6453,  6463,  6473,
        6483,  6493,  8202,  8213,  8223,  8233,  8243,  8253,  8263,
        8273,  8283,  8293,  8412,  8423,  8433,  8443,  8453,  8463,
        9692,  9703,  9713,  9723,  9733,  9743,  9753,  9763,  9773,
        9783,  9842,  9853,  9863,  9873,  9883,  9893,  9903,  9913,
        9923,  9933,  9943,  9953,  9963,  9973,  9983, 12194, 12204,
       12214, 12224, 12234, 12244, 12254, 12264, 12274, 12284, 12294,
       12304, 12344, 12354, 12364, 12374, 12384, 12394, 12404, 12414,
       12424, 12434, 12444, 12454, 12464, 12474, 12484, 14193, 14203,
       14213, 14223, 14233, 14243, 14253, 14263, 14273, 14283, 14293,
       14303, 14313, 14323, 14443, 14453, 14463, 14473, 14483, 14493,
       14503, 14513, 14523, 14533, 14543, 16183, 16193, 16203, 16213,
       16223, 16233, 16243, 16253, 16263, 16273, 16372, 16383, 16393,
       16403, 16413, 16423, 16433, 16443, 16453, 16463, 16473, 16483,
       16493, 16503, 18203, 18213, 18223, 18233, 18243, 18253, 18263,
       18273, 18283, 18293, 18303, 18341, 18352, 18362, 18372, 18382,
       18392, 18402, 18412, 18422, 18432, 19183, 19194, 19204, 19214,
       19224, 19234, 19244, 19254, 19264, 19334, 19344, 19354, 19364,
       19374, 19384, 19394, 19404, 19414, 19424, 19434, 19444, 19454,
       19464, 19474, 20191, 20203, 20213, 20223, 20233, 20243, 20253,
       20263, 20273, 20283, 20351, 20363, 20373, 20383, 20393, 20403,
       20413, 23193, 23203, 23213, 23223, 23233, 23243, 23253, 23263,
       23273, 23283, 23373, 23383, 23393, 23403, 23413, 23423, 23433,
       23443, 23453, 23463, 25182, 25193, 25203, 25213, 25223, 25233,
       25243, 25253, 25263, 25273, 25283, 25293, 25303, 25313, 25323,
       25333, 25343, 25353, 25363, 25433, 25443, 25453, 25463, 25473,
       25483, 25493, 25503, 25513, 25523, 25533, 25543, 25553, 25563,
       25573, 25583, 26771, 26783, 26792, 26801, 26810, 26819, 26828,
       26837, 26846, 27098, 27107, 27116, 27125, 27134, 27143, 27330,
       27341, 27350, 27359, 27368, 27377, 27386, 27395, 27404, 27413]), array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
       104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
       117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
       143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
       156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
       169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
       182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194,
       195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
       208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
       221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233,
       234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246,
       247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259,
       260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272,
       273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285,
       286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298,
       299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
       312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324,
       325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 341, 342, 343,
       344, 345, 346, 347, 348, 349, 335, 336, 337, 338, 339, 340, 350,
       351, 352, 353, 354, 355, 356, 357, 358, 359]))
2019-03-08 13:14:07,993 : DEBUG : np.where(overlaps == gt_max_overlaps)后得到的gt_argmax_overlaps:(360,)
2019-03-08 13:14:07,993 : DEBUG : 每个位置上的9个anchor中overlap最大的认为是前景,都打上前景标签1
2019-03-08 13:14:07,993 : DEBUG : gt_argmax_overlaps:array([ 1829,  1839,  1848,  1857,  1866,  1875,  1884,  1893,  1902,
        1911,  1920,  1964,  1974,  1983,  1992,  2001,  2010,  2019,
        2028,  2037,  2046,  2055,  2064,  2126,  2135,  2144,  2153,
        2162,  2738,  2747,  2756,  2765,  2774,  2783,  2792,  2801,
        2810,  2819,  2828,  2837,  2846,  2918,  2928,  2937,  2946,
        2955,  2964,  2973,  2982,  2991,  3000,  3009,  5202,  5212,
        5222,  5232,  5242,  5252,  5262,  5272,  5282,  5352,  5362,
        5372,  5382,  5392,  5402,  5412,  5422,  5432,  5442,  5452,
        5462,  6183,  6193,  6203,  6213,  6223,  6233,  6243,  6253,
        6263,  6273,  6283,  6293,  6303,  6351,  6363,  6373,  6383,
        6393,  6403,  6413,  6423,  6433,  6443,  6453,  6463,  6473,
        6483,  6493,  8202,  8213,  8223,  8233,  8243,  8253,  8263,
        8273,  8283,  8293,  8412,  8423,  8433,  8443,  8453,  8463,
        9692,  9703,  9713,  9723,  9733,  9743,  9753,  9763,  9773,
        9783,  9842,  9853,  9863,  9873,  9883,  9893,  9903,  9913,
        9923,  9933,  9943,  9953,  9963,  9973,  9983, 12194, 12204,
       12214, 12224, 12234, 12244, 12254, 12264, 12274, 12284, 12294,
       12304, 12344, 12354, 12364, 12374, 12384, 12394, 12404, 12414,
       12424, 12434, 12444, 12454, 12464, 12474, 12484, 14193, 14203,
       14213, 14223, 14233, 14243, 14253, 14263, 14273, 14283, 14293,
       14303, 14313, 14323, 14443, 14453, 14463, 14473, 14483, 14493,
       14503, 14513, 14523, 14533, 14543, 16183, 16193, 16203, 16213,
       16223, 16233, 16243, 16253, 16263, 16273, 16372, 16383, 16393,
       16403, 16413, 16423, 16433, 16443, 16453, 16463, 16473, 16483,
       16493, 16503, 18203, 18213, 18223, 18233, 18243, 18253, 18263,
       18273, 18283, 18293, 18303, 18341, 18352, 18362, 18372, 18382,
       18392, 18402, 18412, 18422, 18432, 19183, 19194, 19204, 19214,
       19224, 19234, 19244, 19254, 19264, 19334, 19344, 19354, 19364,
       19374, 19384, 19394, 19404, 19414, 19424, 19434, 19444, 19454,
       19464, 19474, 20191, 20203, 20213, 20223, 20233, 20243, 20253,
       20263, 20273, 20283, 20351, 20363, 20373, 20383, 20393, 20403,
       20413, 23193, 23203, 23213, 23223, 23233, 23243, 23253, 23263,
       23273, 23283, 23373, 23383, 23393, 23403, 23413, 23423, 23433,
       23443, 23453, 23463, 25182, 25193, 25203, 25213, 25223, 25233,
       25243, 25253, 25263, 25273, 25283, 25293, 25303, 25313, 25323,
       25333, 25343, 25353, 25363, 25433, 25443, 25453, 25463, 25473,
       25483, 25493, 25503, 25513, 25523, 25533, 25543, 25553, 25563,
       25573, 25583, 26771, 26783, 26792, 26801, 26810, 26819, 26828,
       26837, 26846, 27098, 27107, 27116, 27125, 27134, 27143, 27330,
       27341, 27350, 27359, 27368, 27377, 27386, 27395, 27404, 27413])
2019-03-08 13:14:07,996 : DEBUG : overlap大于0.7的认为是前景
2019-03-08 13:14:07,996 : DEBUG : max_overlaps:array([0., 0., 0., ..., 0., 0., 0.])
2019-03-08 13:14:07,996 : DEBUG : cfg.RPN_FG_FRACTION 0 * cfg.RPN_BATCHSIZE 300 = 150
2019-03-08 13:14:07,997 : DEBUG : fg_inds = (558,)
2019-03-08 13:14:07,997 : DEBUG : 只保留num_fg个正样本，剩下的正样本去掉，置成-1
2019-03-08 13:14:08,000 : DEBUG : 开始计算bbox的差：anchors(图内的)和gt_boxes[argmax_overlaps, :] array([0, 0, 0, ..., 0, 0, 0])
2019-03-08 13:14:08,001 : DEBUG : _compute_targets
2019-03-08 13:14:08,001 : DEBUG : ex_rois:anchors:(29150, 4)
2019-03-08 13:14:08,002 : DEBUG : gt_rois:gts:(29150, 5)
2019-03-08 13:14:08,004 : DEBUG : 计算完的bbox regression结果：(29150, 4)
2019-03-08 13:14:08,013 : DEBUG : 最后的这个超长的anchor_target_layer返回结果为：
2019-03-08 13:14:08,013 : DEBUG : rpn_labels:(1, 63, 50, 10)
2019-03-08 13:14:08,013 : DEBUG : rpn_bbox_targets:(1, 63, 50, 40)
2019-03-08 13:14:08,013 : DEBUG : rpn_bbox_inside_weights:(1, 63, 50, 40)
2019-03-08 13:14:08,013 : DEBUG : rpn_bbox_outside_weights:(1, 63, 50, 40)
2019-03-08 13:14:08.027746: I tensorflow/core/kernels/logging_ops.cc:79] rpn_bbox_targets tensor[1 63 50 40]
2019-03-08 13:14:08.027761: I tensorflow/core/kernels/logging_ops.cc:79] rpn_bbox_inside_weights tensor[1 63 50 40]
2019-03-08 13:14:08.027827: I tensorflow/core/kernels/logging_ops.cc:79] rpn_bbox_outside_weights tensor[1 63 50 40]
2019-03-08 13:14:08.028096: I tensorflow/core/kernels/logging_ops.cc:79] rpn_labels tensor[1 63 50 10]
2019-03-08 13:14:08.029985: I tensorflow/core/kernels/logging_ops.cc:79] rpn_cls_score[300 1 2]
2019-03-08 13:14:08.030203: I tensorflow/core/kernels/logging_ops.cc:79] rpn_bbox_targets[300 1 4]
2019-03-08 13:14:08.030354: I tensorflow/core/kernels/logging_ops.cc:79] rpn_label[300 1]
2019-03-08 13:14:08.030386: I tensorflow/core/kernels/logging_ops.cc:79] 我们来看看lstm预测的bbox_pred和anchor_target_layer选出来的anchor组成的bbox_targets的shape：[300 1 4]
2019-03-08 13:14:08.030532: I tensorflow/core/kernels/logging_ops.cc:79] rpn_bbox_pred[300 1 4]
2019-03-08 13:14:08.030769: I tensorflow/core/kernels/logging_ops.cc:79] 做交叉熵了[300 1]

```


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
