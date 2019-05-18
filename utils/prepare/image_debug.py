from PIL import Image, ImageDraw
import os,cv2,shutil
from utils.rpn_msr.anchor_target_layer import anchor_target_layer_process as process_anchor
from utils import init_logger
import argparse,logging
import numpy as np

logger = logging.getLogger("ImageDebugger")


# load 标签坐标
def _load_label(full_label_file_name_path):
    if not (os.path.exists(full_label_file_name_path)): return None

    label_xy = []
    with open(full_label_file_name_path, "r") as label_file:
        for line in label_file:
            if line.strip('\n') == "":
                continue
            # logger.debug("(%s)" % line)
            cord_xy = line.split(",")[0:8]  # 有可能最后一列是标签，所以只取前8列
            label_xy.append([int(float(p)) for p in cord_xy])
    return label_xy


# 如果发现这个bbox的正样本数量不够，就给他移动到 xxx.move目录
def _move_file(orignal_file_full_path):
    if not os.path.exists(orignal_file_full_path):
        logger.error("要移动的文件不存在：%s",orignal_file_full_path)
        return
    dir,name = os.path.split(orignal_file_full_path)
    new_dir = os.path.join(dir,".move")
    if not os.path.exists(new_dir): os.makedirs(new_dir)
    shutil.move(orignal_file_full_path, new_dir)
    logger.debug("移动文件%s=>%s",orignal_file_full_path, new_dir)

# 不用跑起来，直接调用处理，来验证bbox的正样本数量
def anchor_process(image_name,label_name,split_label_name):
    image = cv2.imread(image_name)
    split_label_xys = _load_label(split_label_name)
    W,H = image.shape[:2]
    image_info = np.array(image.shape).reshape([1, 3])
    w,h = round(W/16), round(H/16)
    labels, bbox_targets, _, _ = process_anchor((w,h),np.array(split_label_xys),image_info,[16, ],[16],np.array([image_name]),True)

    num = (labels == 1).sum()
    if num<400:
        logger.debug("bbox正样例<500[%d]，移动图片%s", num, image_name)
        _move_file(image_name)
        _move_file(label_name)
        _move_file(split_label_name)

# /test/abc/=> /test/abc.move
def d(a,b):
    if a[-1]=='/':
        a = a[:-1]
    return a+b


def draw_image(image_name,label_name,split_label_name):
    # 先打开原图
    image = Image.open(image_name)
    draw = ImageDraw.Draw(image)
    # 得到大框和小框的坐标文件数组，注意区别，大框长度是8，小框长度是4
    label_xys = _load_label(label_name)
    logger.debug("%d个大框" , len(label_xys))
    if label_xys:
        # 画一句话最外面大框，8个坐标，画4边型
        for one_img_pos in label_xys:
            cord_xy = [(one_img_pos[0], one_img_pos[1]),
                       (one_img_pos[2], one_img_pos[3]),
                       (one_img_pos[4], one_img_pos[5]),
                       (one_img_pos[6], one_img_pos[7])]
            draw.polygon(cord_xy, outline='red')
    split_label_xys = _load_label(split_label_name)
    logger.debug("%d个小框" , len(split_label_xys))
    if split_label_xys:
        # 画每一个anchor，4个坐标，所以画矩形
        for one_img_pos in split_label_xys:
            draw.rectangle(tuple(one_img_pos), outline='green')
    # 得到要画框后的图片文件的存放路径（大框和小框画到一个文件上）'
    draw_dir = d(image_dir, ".draw")
    if not os.path.exists(draw_dir): os.makedirs(draw_dir)
    draw_image_name = os.path.join(draw_dir, img_name)
    # 把画完的图保存到draw目录
    image.save(draw_image_name)


# 注意：需要在根目录下运行，存到 /data/train目录下
if __name__ == '__main__':

    init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--action")  # 啥类型的数据啊，train/validate/test
    parser.add_argument("--image_dir")
    parser.add_argument("--label_dir")
    parser.add_argument("--label_split_dir")

    args = parser.parse_args()

    action = args.action
    image_dir = args.image_dir
    label_dir = args.label_dir
    label_split_dir = args.label_split_dir

    i = 0
    for img_name in os.listdir(image_dir):

        name, ext = os.path.splitext(img_name)
        if ext.lower() not in ['.jpg', '.png']: continue
        # 得到两个标签的文件名：大框：格式是8个数[x1,y1,x2,y2,x3,y3,x4,y4]，小框：格式是4个数 [x1,y1,x2,y2]
        lab_name = name + ".txt"
        image_name = d(image_dir, img_name)
        label_name = d(label_dir, lab_name)
        split_label_name = d(label_split_dir, lab_name)

        if not os.path.exists(image_name):
            logger.debug("[ERROR]图片不存在%s" , image_name)
            continue
        if not os.path.exists(split_label_name):
            logger.debug("[ERROR]小框标签不存在%s" , split_label_name)
            continue
        if not os.path.exists(label_name):
            logger.debug("[ERROR]大框标签不存在%s" , label_name)
            continue

        if action=="draw": draw_image(image_name,label_name,split_label_name)

        if action=="anchor": anchor_process(image_name,label_name,split_label_name)

        i += 1
        logger.debug("已处理完第%d张图片：[%s]" , i, img_name)