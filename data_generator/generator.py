from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageOps
import random
import cv2
import numpy as np
import os,math
import logging as logger

'''
1. 从文字库随机选择10个字符
2. 生成图片
3. 随机使用函数
'''

# #############
# 设置各种的参数：
# #############

DEBUG=False
ROOT="data_generator"   # 定义运行时候的数据目录，原因是imgen.sh在根部运行
DATA_DIR="data"
MAX_LENGTH=12   # 可能的最大长度（字符数）
MIN_LENGTH=5    # 可能的最小长度（字符数）
MAX_FONT_SIZE = 20 # 最大的字体
MIN_FONT_SIZE = 15 # 最小的字体号
MAX_LINE_HEIGHT= 100   # 最大的高度（像素）
MIN_LINE_HEIGHT= MIN_FONT_SIZE + 12   # 最小的高度（像素）


# 颜色的算法是，产生一个基准，然后RGB上下浮动FONT_COLOR_NOISE
MAX_FONT_COLOR = 100    # 最大的可能颜色
FONT_COLOR_NOISE = 10   # 最大的可能颜色
ONE_CHARACTOR_WIDTH = 1024# 一个字的宽度
ROTATE_ANGLE = 4        # 随机旋转角度
GAUSS_RADIUS_MIN = 0.5  # 高斯模糊的radius最小值
GAUSS_RADIUS_MAX = 0.8  # 高斯模糊的radius最大值

# 之前的设置，太大，我决定改改
# MAX_BACKGROUND_WIDTH = 1600
# MIN_BACKGROUND_WIDTH = 800
# MAX_BACKGROUND_HEIGHT = 2500
# MIN_BACKGROUND_HEIGHT = 1000
MAX_BACKGROUND_WIDTH = 700
MIN_BACKGROUND_WIDTH = 500
MAX_BACKGROUND_HEIGHT = 900
MIN_BACKGROUND_HEIGHT = 800


MIN_BLANK_WIDTH = 50 # 最小的句子间的随机距离
MAX_BLANK_WIDTH = 100 # 最长的句子间距离

INTERFER_LINE_NUM = 10
INTERFER_POINT_NUM = 2000
INTERFER_LINE_WIGHT = 2
INTERFER_WORD_LINE_NUM = 4
INTERFER_WORD_POINT_NUM = 20
INTERFER_WORD_LINE_WIGHT = 1

# 改进图片的质量：
# a、算清楚留白，不多留
# b、旋转一下，不切去变了
# c、增加整张纸的脏变形，暗影，干扰线
# D、造单字的样本
# F、造数字样本 1,000,00.000类似的

# 各种可能性的概率
POSSIBILITY_ROTOATE = 0.4   # 文字的旋转
POSSIBILITY_INTEFER = 0.2   # 需要被干扰的图片，包括干扰线和点
POSSIBILITY_WORD_INTEFER = 0.1 # 需要被干扰的图片，包括干扰线和点
POSSIBILITY_AFFINE  = 0.3   # 需要被做仿射的文字
POSSIBILITY_PURE_NUM = 0.1  # 需要产生的纯数字
POSSIBILITY_DATE = 0.1      # 需要产生的纯数字
POSSIBILITY_SINGLE = 0.01   # 单字的比例

# # 测试用
# POSSIBILITY_ROTOATE = 1   # 文字的旋转
# POSSIBILITY_INTEFER = 0   # 需要被干扰的图片，包括干扰线和点
# POSSIBILITY_WORD_INTEFER = 0   # 需要被干扰的图片，包括干扰线和点
# POSSIBILITY_AFFINE  = 0   # 需要被做仿射的文字
# POSSIBILITY_PURE_NUM = 0  # 需要产生的纯数字
# POSSIBILITY_DATE = 0      # 需要产生的纯数字
# POSSIBILITY_SINGLE = 0    # 单字的比例

MAX_GENERATE_NUM = 1000000000

# 仿射的倾斜的错位长度  |/_/, 这个是上边或者下边右移的长度
AFFINE_OFFSET = 12

def _get_random_point(x_scope,y_scope):
    x1 = random.randint(0,x_scope)
    y1 = random.randint(0,y_scope)
    return x1,y1

# 加载字符集，charset.txt，最后一个是空格
# 为了兼容charset.txt和charset6k.txt，增加鲁棒性，改一下
# 先读入内存，除去
def _get_charset(charset_file):
    charset = open(charset_file, 'r', encoding='utf-8').readlines()
    charset = [ch.strip("\n") for ch in charset]
    charset = "".join(charset)
    charset = list(charset)
    if charset[-1]!=" ":
        charset.append(" ")
    return charset


# 画干扰线
def randome_intefer_line(img,possible,line_num,weight):

    if not _random_accept(possible): return

    w,h = img.size
    draw = ImageDraw.Draw(img)
    line_num = random.randint(0, line_num)

    for i in range(line_num):
        x1,y1 = _get_random_point(w,h)
        x2, y2 = _get_random_point(w,h)
        _weight = random.randint(0, weight)
        draw.line([x1,y1,x2,y2],_get_random_color(),_weight)

    del draw

# 画干扰点
def randome_intefer_point(img,possible,num):

    if not _random_accept(possible): return

    w,h = img.size
    draw = ImageDraw.Draw(img)

    point_num = random.randint(0, num)
    for i in range(point_num):
        x,y = _get_random_point(w,h)
        draw.point([x,y], _get_random_color())
    del draw

# 专门用来产生数字，可能有负数
def _generate_num():
    num = random.randint(-MAX_GENERATE_NUM,MAX_GENERATE_NUM)
    # print(num)
    need_format = random.choice([True,False])

    if (need_format):
        return "{:,}".format(num)

    return str(num)


# 专门用来产生日期，各种格式的
def _generate_date():
    import time
    now = time.time()

    date_formatter = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y年%m月%d日",
        "%Y%m%d ",
        "%y-%m-%d ",
        "%y/%m/%d ",
        "%y年%m月%d日",
        "%Y%m%d "
    ]

    _format = random.choice(date_formatter)

    _timestamp = random.uniform(0,now)

    time_local = time.localtime(_timestamp)

    return time.strftime(_format, time_local)

# 随机生成文字，长度是10-30个之间，512像素
def _generate_words(charset):
    length = random.randint(MIN_LENGTH,MAX_LENGTH)
    s = ""
    for i in range(length):
        j = random.randint(0, len(charset) - 1)
        s += charset[j]
    if DEBUG: print("随机生成的汉字字符串[%s]，%d" %(s,length))
    return s,length


# 从文字库中随机选择n个字符
def _get_random_text(charset):

    # 产生随机数字
    if _random_accept(POSSIBILITY_PURE_NUM):
        s_num = _generate_num()
        return s_num,len(s_num)

    # 产生随机日期
    if _random_accept(POSSIBILITY_DATE):
        s_date = _generate_date()
        return s_date, len(s_date)

    # start = random.randint(0, len(info_str)-MAX_LENGTH-1)
    # length = random.randint(MIN_LENGTH, MAX_LENGTH)
    #
    # # 是否产生单字
    # if _random_accept(POSSIBILITY_SINGLE):length = 1
    #
    # end = start + length
    # random_word = info_str[start:end]
    # if DEBUG: print("截取内容[%s]，%d" %(random_word,length))
    # import re
    # rex = re.compile(' ')
    # random_word = rex.sub('', random_word)
    return _generate_words(charset)

# 产生随机颜色
def _get_random_color():
    base_color = random.randint(0, MAX_FONT_COLOR)
    noise_r = random.randint(0, FONT_COLOR_NOISE)
    noise_g = random.randint(0, FONT_COLOR_NOISE)
    noise_b = random.randint(0, FONT_COLOR_NOISE)

    noise = np.array([noise_r,noise_g,noise_b])
    font_color = (np.array(base_color) + noise).tolist()

    return tuple(font_color)

# 生成一张图片, 1200x1920
def load_all_backgroud_images(bground_path):
    bground_list = []

    for img_name in os.listdir(bground_path):
        image = Image.open(bground_path + img_name)
        if image.mode == "L":
            logger.error("图像[%s]是灰度的，转RGB",img_name)
            image = image.convert("RGB")

        bground_list.append(image)
        logger.debug("    加载背景图片：%s",bground_path + img_name)
    logger.debug("所有图片加载完毕")

    return bground_list

# 生成一张背景图，大小随机
def create_backgroud_image(bground_list):

    # 从背景目录中随机选一张"纸"的背景
    bground_choice = random.choice(bground_list)

    # return image, width, height
    return random_image_size(bground_choice)

def _add_noise(img):
    # img = (scipy.misc.imread(filename)).astype(float)
    noise_mask = np.random.poisson(img)
    noisy_img = img + noise_mask
    return noisy_img

# 模糊函数
def random_blur(image,font_size):
    # 随机选取模糊参数
    radius = random.uniform(GAUSS_RADIUS_MIN,GAUSS_RADIUS_MAX)
    filter_ = random.choice(
                            [
                             # ImageFilter.SMOOTH, 太模糊
                             # ImageFilter.DETAIL, 太模糊
                             ImageFilter.GaussianBlur(radius=radius),
                             # ImageFilter.EDGE_ENHANCE,      #边缘增强滤波,这个效果不好，暂时注释掉，边缘扣掉的太多
                             ImageFilter.SHARPEN
                             ]) #为深度边缘增强滤波


    if DEBUG: print("模糊函数：%s" % str(filter_))


    image = image.filter(filter_)
    return image

# 随机裁剪图片的各个部分
def random_image_size(image):

    while True:
        # 产生随机的大小
        height = random.randint(MIN_BACKGROUND_HEIGHT,MAX_BACKGROUND_HEIGHT)
        width = random.randint(MIN_BACKGROUND_WIDTH,MAX_BACKGROUND_WIDTH)

        # 高度和宽度随机后，还要随机产生起始点x,y，但是要考虑切出来不能超过之前纸张的大小，所以做以下处理：
        size = image.size
        x_scope =  size[0] - width
        y_scope =  size[1] - height
        if x_scope<0: continue
        if y_scope<0: continue
        x = random.randint(0,x_scope)
        y = random.randint(0,y_scope)

        # logger.debug("产生随机的图片宽[%d]高[%d]",width,height)
        image = image.crop((x,y,x+width,y+height))
        # logger.debug("剪裁图像:x=%d,y=%d,w=%d,h=%d",x,y,width,height)
        return image, width, height

# 随机接受概率
def _random_accept(accept_possibility):
    return np.random.choice([True,False], p = [accept_possibility,1 - accept_possibility])

# 旋转函数
def random_rotate(img,points):
    ''' ______________
        |  /        /|
        | /        / |
        |/________/__|
        旋转可能有两种情况，一种是矩形，一种是平行四边形，
        但是传入的points，就是4个顶点，
    '''
    if not _random_accept(POSSIBILITY_ROTOATE): return img,points # 不旋转

    w,h = img.size

    center = (w//2,h//2)

    if DEBUG: print("需要旋转")
    degree = random.uniform(-ROTATE_ANGLE, ROTATE_ANGLE)  # 随机旋转0-8度
    if DEBUG: print("旋转度数:%f" % degree)
    return img.rotate(degree,center=center,expand=1),_rotate_points(points,center,degree)


# 随机仿射一下，也就是歪倒一下
# 不能随便搞，我现在是让图按照平行方向歪一下，高度不变，高度啊，大小啊，靠别的控制，否则，太乱了
def random_affine(img):

    HEIGHT_PIX = 10
    WIDTH_PIX = 50

    # 太短的不考虑了做变换了
    # print(img.size)
    original_width = img.size[0]
    original_height = img.size[1]
    points = [(0,0), (original_width,0), (original_width,original_height), (0,original_height)]

    if original_width<WIDTH_PIX: return img,points
    # print("!!!!!!!!!!")
    if not _random_accept(POSSIBILITY_AFFINE): return img,points

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)

    is_top_fix = random.choice([True,False])

    bottom_offset = random.randint(0,AFFINE_OFFSET) # bottom_offset 是 上边或者下边 要位移的长度

    height = img.shape[0]

    # 这里，我们设置投影变换的3个点的原则是，使用    左上(0,0)     右上(WIDTH_PIX,0)    左下(0,HEIGHT_PIX)
    # 所以，他的投影变化，要和整个的四边形做根据三角形相似做换算
    # .
    # |\
    # | \
    # |__\  <------投影变化点,  做三角形相似计算，offset_ten_pixs / bottom_offset =  HEIGHT_PIX / height
    # |   \                   所以： offset_ten_pixs = (bottom_offset * HEIGHT_PIX) / height
    # |____\ <-----bottom_offset
    offset_ten_pixs = int(HEIGHT_PIX * bottom_offset / height)   # 对应10个像素的高度，应该调整的横向offset像素
    width = int(original_width  + bottom_offset )#

    pts1 = np.float32([[0, 0], [WIDTH_PIX, 0], [0, HEIGHT_PIX]])  # 这就写死了，当做映射的3个点：左上角，左下角，右上角


    #\---------\
    # \         \
    #  \_________\
    if is_top_fix:  # 上边固定，意味着下边往右
        # print("上边左移")
        pts2 = np.float32([[0, 0], [WIDTH_PIX, 0], [offset_ten_pixs, HEIGHT_PIX]])  # 看，只调整左下角
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (width, height))
        points = [(0,0),
                  (original_width,0),
                  (width,original_height),
                  (bottom_offset,original_height)]
    #  /---------/
    # /         /
    #/_________/
    else:  # 下边固定，意味着上边往右
        # 得先把图往右错位，然后
        # 先右移
        # print("上边右移")
        H = np.float32([[1, 0, bottom_offset], [0, 1, 0]])  #
        img = cv2.warpAffine(img, H, (width, height))
        # 然后固定上部，移动左下角
        pts2 = np.float32([[0, 0], [WIDTH_PIX, 0], [-offset_ten_pixs, HEIGHT_PIX]])  # 看，只调整左下角
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, (width, height))
        points = [(bottom_offset,0),
                  (original_width+bottom_offset,0),
                  (width,original_height),
                  (0,original_height)]


    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

    return img,points

# 得到一个随机大小的字体大小
def random_font_size():
    font_size = random.randint(MIN_FONT_SIZE,MAX_FONT_SIZE)
    return font_size

# 从目录中随机选择一种字体
def random_font(font_path):
    font_list = os.listdir(font_path)
    random_font = random.choice(font_list)

    return font_path + random_font

# 产生一张图的所有的文字
#
#
def generate_all(bground_list, image_name,label_name,charset):

    # 先创建一张图，宽度和高度都是随机的
    image, w, h = create_backgroud_image(bground_list)


    # 在整张图上产生干扰点和线
    randome_intefer_line(image,POSSIBILITY_INTEFER,INTERFER_LINE_NUM,INTERFER_LINE_WIGHT)
    randome_intefer_point(image,POSSIBILITY_INTEFER,INTERFER_POINT_NUM)

    # 上面先空出一行来
    y = random.randint(MIN_LINE_HEIGHT, MAX_LINE_HEIGHT)
    i = 0

    # 一张图的坐标，格式是[[x1,y1,x2,y2,x3,y3,x4,y4],....],8个，这个不是我发明的，是这个开源项目要求的
    # 后续要用这个8个坐标去产生更小的anchor
    one_image_labels = []

    while True:
        # 产生一行的标签,格式是[x1,y1,x2,y2,x3,y3,x4,y4]
        one_row_labels = generate_row( i, y ,background_image = image, image_width=w,charset=charset)
        one_image_labels += one_row_labels

        # 主要是提前看看高度是否超过底边了，超过了，就算产生完了
        line_height = random.randint(MIN_LINE_HEIGHT,MAX_LINE_HEIGHT)
        # logger.debug("行高：%d",line_height)
        y = y + line_height
        i += 1
        if y + line_height > h: # 多算一行高度，来看看是否到底部了？
            # logger.debug("达到了图片最高的位置")
            break

    image.save(image_name)
    logger.debug("生成样本图像  %s",image_name)

    with open(label_name,"w") as label_file:
        for label in one_image_labels:
            xy_info = ",".join([str(pos) for pos in label])
            label_file.write(xy_info)
            label_file.write("\n")

# 产生一行的文字，里面又有一个循环，是产生多个句子
def generate_row(i,y, background_image,image_width,charset):
    # logger.debug("---------------------------------")
    # logger.debug("开始准备产生第%d行", i)
    next_x = 0

    one_row_labels = []

    # 先随机空出一部分
    blank_width = random.randint(MIN_BLANK_WIDTH, MAX_BLANK_WIDTH)
    next_x += blank_width
    while True:

        # 产生一个句子，并返回这个句子的文字，还有这个句子句尾巴的x坐标，如果是None，说明已经到最右面了，也就是说，这行生成结束了
        next_x,label = process_one_sentence(next_x, y,background_image,image_width,charset)
        if next_x is None: break

        one_row_labels.append(label) # 把产生的4个坐标加入到这张图片的标签数组中

        # 产生一个句子句子之间的空隙
        blank_width = random.randint(MIN_BLANK_WIDTH,MAX_BLANK_WIDTH)

        # logger.debug("空格长度%d",blank_width)
        next_x += blank_width

    return one_row_labels
    # logger.debug("第%d行图片产生完毕",i)

# 因为英文、数字、符号等ascii可见字符宽度短，所以要计算一下他的实际宽度，便于做样本的时候的宽度过宽
def caculate_text_shape(text,font):

    #获得文字的offset位置
    offsetx, offsety = font.getoffset(text)

    #获得文件的大小
    width, height=font.getsize(text)

    width = width #- offsetx
    height = height #- offsety

    return width,height

def _rotate_one_point(xy, center, theta):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    cord = (
        # (xy[0] - center[0]) * cos_theta - (xy[1]-center[1]) * sin_theta + xy[0],
            # (xy[0] - center[0]) * sin_theta + (xy[1]-center[1]) * cos_theta + xy[1]
            (xy[0] - center[0]) * cos_theta - (xy[1] - center[1]) * sin_theta + center[0],
            (xy[0] - center[0]) * sin_theta + (xy[1] - center[1]) * cos_theta + center[1]

        )
    # print("旋转后的坐标：")
    # print(cord)
    return cord


def _rotate_points(points,center, degree):
    theta = math.radians(-degree)


    original_min_x, original_min_y = np.array(points).max(axis=0)

    rotated_points = [_rotate_one_point(xy, center, theta) for xy in points]

    rotated_min_x, rotated_min_y = np.array(rotated_points).max(axis=0)

    x_offset = abs(rotated_min_x - original_min_x)
    y_offset = abs(rotated_min_y - original_min_y)

    rotated_points = [(xy[0]+x_offset, xy[1]+y_offset) for xy in rotated_points]

    return rotated_points

def create_one_sentence_image(charset):
    # 随机选取10个字符，是从info.txt那个词库里，随机挑的长度的句子
    random_word,length = _get_random_text(charset)

    # 字号随机
    font_size = random_font_size()
    # 随机选取字体大小、颜色、字体
    font_name = random_font(ROOT+'/font/')
    font_color = _get_random_color()
    font = ImageFont.truetype(font_name, font_size)
    # logger.debug("字体的颜色是：[%r]",font_color)

    # 因为有数字啊，英文字母啊，他的宽度比中文短，所以为了框套的更严丝合缝，好好算算宽度
    width, height = caculate_text_shape(random_word.strip(),font)


    words_image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(words_image)
    # 注意下，下标是从0,0开始的，是自己的坐标系
    draw.text((0, 0), random_word, fill=font_color, font=font)

    ############### PIPELINE ###########################

    words_image,points = random_affine(words_image)
    words_image,points = random_rotate(words_image,points)
    words_image = random_blur(words_image,font_size)
    randome_intefer_line(words_image, POSSIBILITY_WORD_INTEFER,INTERFER_WORD_LINE_NUM,INTERFER_WORD_LINE_WIGHT)           # 给单个文字做做干扰
    randome_intefer_point(words_image,POSSIBILITY_WORD_INTEFER,INTERFER_WORD_POINT_NUM)
    ############### PIPELINE ###########################

    return words_image,width,height,random_word,points#<----points不一定在是width/height组成的矩形里面了，做仿射和旋转了嘛

# 生成的最小颗粒度，生成一句话
def process_one_sentence(x, y, background_image, image_width,charset):
    words_image, width, _, _, points = create_one_sentence_image(charset)

    # 一算，句子的宽度超了，得嘞，这行咱就算生成完毕
    if x + width > image_width:
        # logger.debug("生成句子的右侧位置[%d]超过行宽[%d]，此行终结", x+words_image_width, image_width)
        return None,None

    background_image.paste(words_image, (x,y), words_image)

    logger.debug("产生了一个句子[%s],坐标(%r)", random_word, points)
    # x1, y1, x2, y2, x3, y3, x4, y4
    label = [
        int(x + points[0][0]), int(y + points[0][1]),
        int(x + points[1][0]), int(y + points[1][1]),
        int(x + points[2][0]), int(y + points[2][1]),
        int(x + points[3][0]), int(y + points[3][1])]

    logger.debug("粘贴到背景上的坐标：(%r)", label)
    return x + width ,label


def init_logger():
    logger.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logger.INFO,
        handlers=[logger.StreamHandler()])


# 注意：需要在根目录下运行，存到 /data/train目录下
if __name__ == '__main__':

    import argparse

    init_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--type") # 啥类型的数据啊，train/validate/test
    parser.add_argument("--dir")  # 这个程序的主目录
    parser.add_argument("--num")  # 生成多少个啊？

    args = parser.parse_args()

    DATA_DIR = args.dir
    TYPE= args.type

    # 处理具有工商信息语义信息的语料库，去除空格等不必要符号
    with open(ROOT+'/info.txt', 'r', encoding='utf-8') as file:
        info_list = [part.replace('\t', '') for part in file.readlines()] # \t不能显示正常，删掉
        info_str = ''.join(info_list)

    total = int(args.num)

    # 生成的图片存放目录
    data_images_dir = os.path.join(DATA_DIR,TYPE,"images")

    # 生成的图片对应的标签的存放目录，注意这个是大框，后续还会生成小框，即anchor，参见split_label.py
    data_labels_dir = os.path.join(DATA_DIR,TYPE,"labels")

    if not os.path.exists(data_images_dir): os.makedirs(data_images_dir)
    if not os.path.exists(data_labels_dir): os.makedirs(data_labels_dir)

    charset = _get_charset("../crnn/charset6k.txt")

    # 预先加载所有的纸张背景
    all_bg_images = load_all_backgroud_images(os.path.join(ROOT,'background/'))

    for num in range(0,total):

        image_name = os.path.join(data_images_dir,str(num)+".png")
        label_name = os.path.join(data_labels_dir,str(num)+".txt")

        generate_all(all_bg_images,image_name,label_name,charset)
        logger.info("已产生[%s]",image_name)


