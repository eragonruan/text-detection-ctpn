from PIL import Image, ImageDraw
import os

'''
    这个程序用来，把生成的句子的外面的大框画出来（红色），
    然后，在把小框（anchor）画出来（绿色），
    
    目录是这么约定的：
    生成的原图：    images
    大框标签：      labels
    anchor标签：   split
    画框的图：      draw
    
    另外，
    大框格式是：[x1,y1,x2,y2,x3,y3,x4,y4]，4个点，所以可以是不规则四边形
    小框格式是：[x1,y1,x2,y2] 2个点，只能是矩形
'''

# load 标签坐标
def _load_label(full_label_file_name_path):
    label_xy = []
    with open(full_label_file_name_path, "r") as label_file:
        for line in label_file:
            label_xy.append([int(p) for p in line.split(",")])
    return label_xy


# 注意：需要在根目录下运行，存到 /data/train目录下
if __name__ == '__main__':

    TYPE= "train"

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type") # 啥类型的数据啊，train/validate/test
    parser.add_argument("--dir")  # 这个程序的主目录

    args = parser.parse_args()

    data_dir = args.dir
    type = args.type


    # 原图目录
    data_images_dir = os.path.join(data_dir,type,"images")

    # 大框标签目录（坐标）
    data_labels_dir = os.path.join(data_dir,type,"labels")

    # 保存小框标签（坐标）的目录
    data_split_labels_dir = os.path.join(data_dir,type,"split")

    # 要画出来的图片存放的目录
    data_draws_dir = os.path.join(data_dir, type, "draws")


    if not os.path.exists(data_draws_dir): os.makedirs(data_draws_dir)

    i = 0
    for img_name in os.listdir(data_images_dir):

        name, ext = os.path.splitext(img_name)
        if ext.lower() not in ['.jpg', '.png']: continue

        # 得到两个标签的文件名：大框：格式是8个数[x1,y1,x2,y2,x3,y3,x4,y4]，小框：格式是4个数 [x1,y1,x2,y2]
        split_lab_name = lab_name = name+".txt"

        # 得到原图
        image_name = os.path.join(data_images_dir, img_name)

        # 得到标签的full path，带目录的全路径
        label_name          = os.path.join(data_labels_dir, lab_name)
        split_label_name    = os.path.join(data_split_labels_dir,lab_name)

        # 得到要画框后的图片文件的存放路径（大框和小框画到一个文件上）
        draw_image_name    = os.path.join(data_draws_dir, img_name)

        # 得到大框和小框的坐标文件数组，注意区别，大框长度是8，小框长度是4
        label_xys = _load_label(label_name)
        split_label_xys = _load_label(split_label_name)

        # 先打开原图
        image = Image.open(image_name)
        draw = ImageDraw.Draw(image)

        # 画一句话最外面大框，8个坐标，画4边型
        for one_img_pos in label_xys:
            cord_xy = [(one_img_pos[0],one_img_pos[1]),
                       (one_img_pos[2], one_img_pos[3]),
                       (one_img_pos[4], one_img_pos[5]),
                       (one_img_pos[6], one_img_pos[7])]
            draw.polygon(cord_xy, outline='red')

        # 画每一个anchor，4个坐标，所以画矩形
        for one_img_pos in split_label_xys:
            draw.rectangle(tuple(one_img_pos), outline='green')

        # 把画完的图保存到draw目录
        image.save(draw_image_name)

        i+=1
        print("已绘制完第%d张：[%s]"  % (i,draw_image_name))

