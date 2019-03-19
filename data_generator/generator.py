from PIL import Image, ImageDraw, ImageFont, ImageFilter,ImageOps
import random
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
MAX_FONT_SIZE = 30 # 最大的字体
MIN_FONT_SIZE = 20 # 最小的字体号
MAX_LINE_HEIGHT= 100   # 最大的高度（像素）
MIN_LINE_HEIGHT= MIN_FONT_SIZE + 12   # 最小的高度（像素）


# 颜色的算法是，产生一个基准，然后RGB上下浮动FONT_COLOR_NOISE
MAX_FONT_COLOR = 100    # 最大的可能颜色
FONT_COLOR_NOISE = 10   # 最大的可能颜色
ONE_CHARACTOR_WIDTH = 1024# 一个字的宽度
ROTATE_ANGLE = 3        # 随机旋转角度
ROTATE_POSSIBLE = 0.4   # 按照定义的概率比率进行旋转，也就是100张里可能有多少个发生旋转
GAUSS_RADIUS_MIN = 0.8  # 高斯模糊的radius最小值
GAUSS_RADIUS_MAX = 1.3  # 高斯模糊的radius最大值

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


# 改进图片的质量：
# a、算清楚留白，不多留
# b、旋转一下，不切去变了
# c、增加整张纸的脏变形，暗影，干扰线
# D、造单字的样本
# F、造数字样本 1,000,00.000类似的



# 从文字库中随机选择n个字符
def sto_choice_from_info_str():
    start = random.randint(0, len(info_str)-MAX_LENGTH-1)
    length = random.randint(MIN_LENGTH,MAX_LENGTH)
    end = start + length
    random_word = info_str[start:end]
    if DEBUG: print("截取内容[%s]，%d" %(random_word,length))
    return random_word,length

# 产生随机颜色
def random_word_color():
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
        bground_list.append(Image.open(bground_path + img_name))
        logger.debug("    加载背景图片：%s",bground_path + img_name)
    logger.debug("所有图片加载完毕")

    return bground_list

# 生成一张背景图，大小随机
def create_backgroud_image(bground_list):

    # 从背景目录中随机选一张"纸"的背景
    bground_choice = random.choice(bground_list)

    # return image, width, height
    return random_image_size(bground_choice)


def add_noise(img):
    # img = (scipy.misc.imread(filename)).astype(float)
    noise_mask = np.random.poisson(img)
    noisy_img = img + noise_mask
    return noisy_img

# 模糊函数
def random_blur(image):
    # 随机选取模糊参数
    radius = random.uniform(GAUSS_RADIUS_MIN,GAUSS_RADIUS_MAX)
    filter_ = random.choice(
                            [ImageFilter.SMOOTH,
                            ImageFilter.DETAIL,
                            ImageFilter.GaussianBlur(radius=radius),
                            ImageFilter.EDGE_ENHANCE,      #边缘增强滤波 
                            ImageFilter.SHARPEN]) #为深度边缘增强滤波
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


# 旋转函数
def random_rotate(image):

    #按照5%的概率旋转
    rotate = np.random.choice([True,False], p = [ROTATE_POSSIBLE,1 - ROTATE_POSSIBLE])
    if not rotate:
        return image
    if DEBUG: print("需要旋转")
    degree = random.uniform(-ROTATE_ANGLE, ROTATE_ANGLE)  # 随机旋转0-8度
    image = image.rotate(degree)

    return image

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
def generate_all(bground_list, image_name,label_name):

    # 先创建一张图，宽度和高度都是随机的
    image, w, h = create_backgroud_image(bground_list)

    # 上面先空出一行来
    y = random.randint(MIN_LINE_HEIGHT, MAX_LINE_HEIGHT)
    i = 0

    # 一张图的坐标，格式是[[x1,y1,x2,y2,x3,y3,x4,y4],....],8个，这个不是我发明的，是这个开源项目要求的
    # 后续要用这个8个坐标去产生更小的anchor
    one_image_labels = []

    while True:
        # 产生一行的标签,格式是[x1,y1,x2,y2,x3,y3,x4,y4]
        one_row_labels = generate_row( i, y ,background_image = image, image_width=w)
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
def generate_row(i,y, background_image,image_width):
    # logger.debug("---------------------------------")
    # logger.debug("开始准备产生第%d行", i)
    next_x = 0

    one_row_labels = []

    # 先随机空出一部分
    blank_width = random.randint(MIN_BLANK_WIDTH, MAX_BLANK_WIDTH)
    next_x += blank_width
    while True:

        # 产生一个句子，并返回这个句子的文字，还有这个句子句尾巴的x坐标，如果是None，说明已经到最右面了，也就是说，这行生成结束了
        next_x,label = process_one_sentence(next_x, y,background_image,image_width)
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

    width = width - offsetx
    height = height - offsety

    return width,height


# 生成的最小颗粒度，生成一句话
def process_one_sentence(x, y, background_image, image_width):

    # 随机选取10个字符，是从info.txt那个词库里，随机挑的长度的句子
    random_word,length = sto_choice_from_info_str()

    # 字号随机
    font_size = random_font_size()
    # 随机选取字体大小、颜色、字体
    font_name = random_font(ROOT+'/font/')
    font_color = random_word_color()
    font = ImageFont.truetype(font_name, font_size)
    # logger.debug("字体的颜色是：[%r]",font_color)

    # 因为有数字啊，英文字母啊，他的宽度比中文短，所以为了框套的更严丝合缝，好好算算宽度
    width, height = caculate_text_shape(random_word.strip(),font)

    # 一算，句子的宽度超了，得嘞，这行咱就算生成完毕
    if x + width > image_width:
        # logger.debug("生成句子的右侧位置[%d]超过行宽[%d]，此行终结", x+words_image_width, image_width)
        return None,None

    words_image = Image.new('RGBA', (width, height))
    draw = ImageDraw.Draw(words_image)
    draw.text((0, 0), random_word, fill=font_color, font=font)
    w = words_image.rotate(17.5, expand=1)
    background_image.paste(ImageOps.colorize(w, (0, 0, 0), (255, 255, 0)), (242, 60), w)

    # # 生成一个文字图片
    # words_image = Image.new('RGBA', (width, height),(255,255,255,0)) # 假设字是方的，宽+10，高+4个像素
    # draw = ImageDraw.Draw(words_image)
    #
    #
    # # 把这个句子画到生成的文字图片上
    # draw.text((0,0), random_word, fill=font_color, font=font) # TODO???
    #
    # # 旋转之
    # words_image = random_rotate(words_image)
    # # 模糊、锐化之
    # words_image = random_blur(words_image)
    #
    # # 把这个文字图片，贴到背景图片上
    # bans = words_image.split()
    # background_image.paste(words_image, (x, y),mask=bans[3]) # 必须是3，感觉是通道的意思？不理解？？？


    # x1, y1, x2, y2, x3, y3, x4, y4
    label = [x, y,
             x + width, y ,
             x + width, y + height,
             x , y + height]

    # logger.debug("产生了一个句子[%s],坐标(%d,%d)",random_word,x,y)
    return x + width ,label


def init_logger():
    logger.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logger.DEBUG,
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

    # 预先加载所有的纸张背景
    all_bg_images = load_all_backgroud_images(os.path.join(ROOT,'background/'))

    for num in range(0,total):

        image_name = os.path.join(data_images_dir,str(num)+".png")
        label_name = os.path.join(data_labels_dir,str(num)+".txt")

        generate_all(all_bg_images,image_name,label_name)


