import numpy as np
import argparse
import os
from glob import glob
import scipy.misc

print('ssss')
parser = argparse.ArgumentParser()
# 数据集存储位置
parser.add_argument("--dataset_dir", type=str, default='/disks/disk1/guohao/dataset/zhly/kitti_raw', help="where the dataset is stored")
# 数据集转储位置
parser.add_argument("--dump_root", type=str, default='/disks/disk2/guohao/qjh_use/dump_data', help="Where to dump the data")
# 训练序列长度
parser.add_argument("--seq_length", type=int, default=3, help="Length of each training sequence")
# 图片参数
parser.add_argument("--img_height", type=int, default=256, help="image height")
parser.add_argument("--img_width", type=int, default=832, help="image width")

args = parser.parse_args()

def concat_seq(seq):
    for i, im in enumerate(seq):
        if i == 0:
            res = im
        else:
            res = np.hstack((res, im))
    return res

def deal_for_folder(folder_name):
    folder_path = os.path.join(args.dataset_dir, folder_name)
    images_path = glob(folder_path + '/*.jpg')
    images_path.sort()
    cams_path = os.path.join(folder_path, 'cam.txt')
    with open(cams_path, 'r') as f:
        cams = f.readline()
    length = len(images_path)
    seq = [[] for i in range(length - args.seq_length + 1)]
    for i, image_path in enumerate(images_path):
        if i + args.seq_length <= length:
            for j in range(args.seq_length):
                seq[i].append(images_path[i+j])
    imgs = np.zeros([len(seq), args.img_height, args.img_width * args.seq_length, 3])
    for i, se in enumerate(seq):
        im_seq = []
        for _ in se:
            im = scipy.misc.imread(_)
            im_seq.append(im)
        img = concat_seq(im_seq)
        imgs[i] = img
    dump_path = os.path.join(args.dump_root, folder_name)
    for i in range(imgs.shape[0]):
        img_path = os.path.join(dump_path, '%.10d.jpg' % i)
        cam_path = os.path.join(dump_path, '%.10dcam.txt' % i)
        try:
            os.makedirs(dump_path)
        except:
            pass
        scipy.misc.imsave(img_path, imgs[i].astype(np.uint8))
        with open(cam_path, 'w') as f:
            f.writelines(cams)
    print(folder_name + ' has done')

def deals(train_or_test):
    with open(os.path.join(args.dataset_dir, train_or_test + '.txt'), 'r') as f:
        folder_names = f.readlines()
    folder_names = [folder_name[:-1] for folder_name in folder_names]
    for folder_name in folder_names:
        deal_for_folder(folder_name)

def main():
    if not os.path.exists(args.dump_root):
        os.makedirs(args.dump_root)
    deals('train')
    deals('val')
    print('transform over')

if __name__ == '__main__':
    main()


