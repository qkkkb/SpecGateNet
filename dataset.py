import glob
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from utils import get_label_info, one_hot_it

# import sys
# sys.path.append('..')
'''# 
image transform:
    input_transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([,,],[,,]),])
'''


class HyDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, csv_path, scale, mode="train", aux_bands=None, aux_concat=False):
        super().__init__()
        if aux_bands is None:
            aux_bands = []
        self.aux_concat = aux_concat
        self.mode = mode
        self.use_auxbands = (len(aux_bands) > 0)
        img_root_path = os.path.join(data_root, 'img')
        label_root_path = os.path.join(data_root, 'label')

        self.image_list = glob.glob(os.path.join(img_root_path, '*.png'))
        self.label_list = glob.glob(os.path.join(label_root_path, '*.png'))
        self.image_list.sort()
        self.label_list.sort()
        if self.use_auxbands:
            self.image_aux_dict = {}
            for band in aux_bands:
                tmp = glob.glob(os.path.join(img_root_path, 'band_' + band, '*.png'))
                tmp.sort()
                self.image_aux_dict[band] = tmp
        try:
            assert (len(self.image_list) == len(self.label_list))
        except AssertionError as AE:
            print("Sample and Label not match!")
        ############*********Randomize the order of pictures and labels**********#############
        # random.seed(32)
        # random.shuffle(self.image_list)
        # random.seed(32)
        # random.shuffle(self.label_list)  # 使图片和标签打乱顺序后依旧匹配

        self.label_info = get_label_info(csv_path)
        # self.normalization=transforms.Normalize([0.4932640469307336,0.46315285088307845,0.5063463860086066],[0.253050951713527, 0.2611631997437442,0.2732297308757342])
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)
        self.resize_label = transforms.Resize(scale, Image.NEAREST)
        self.to_tensor = transforms.ToTensor()  # 把一个取值范围是[0,255]的PIL.Image或者shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]，取值范围是[0,1.0]的torch.FloadTensor

    def __getitem__(self, index):

        # RGB image
        img = Image.open(self.image_list[index])
        ########image normalization############
        img = self.resize_img(img)
        img = np.array(img)

        # aux bans image
        img_aux = torch.Tensor()
        # if self.use_auxbands:
        #     aux_img_np = []
        #     # k: band name v: images list of the band
        #     for k, v in self.image_aux_dict.items():
        #         tmp = Image.open(v[index]).convert('L')
        #         tmp = self.resize_img(tmp)
        #         aux_img_np.append(np.array(tmp))
        #     aux_img_np = np.stack(aux_img_np, axis=0)
        #     img_aux = self.to_tensor(np.transpose(aux_img_np, (1, 2, 0))).float()

        if self.use_auxbands:
            if self.aux_concat:
                aux_concat_tmp = []
                for k, v in self.image_aux_dict.items():
                    # 8bit
                    tmp = Image.open(v[index]).convert('L')
                    tmp = self.resize_img(tmp)
                    aux_concat_tmp.append(np.array(tmp))
                aux_concat_tmp = np.transpose(np.stack(aux_concat_tmp), (1, 2, 0))
                img = np.dstack((img, aux_concat_tmp))
                pass

            else:
                aux_img_np = []
                # k: band name v: images list of the band
                for k, v in self.image_aux_dict.items():
                    # 8bit
                    tmp = Image.open(v[index]).convert('L')
                    tmp = self.resize_img(tmp)
                    aux_img_np.append(np.array(tmp))

                aux_img_np = np.stack(aux_img_np, axis=0)
                img_aux = self.to_tensor(np.transpose(aux_img_np, (1, 2, 0))).float()

                # 16bit
                # 定义转换
                # image = Image.open(v[index])
                # image = self.resize_img(image)
                # transform = transforms.Compose([
                #     transforms.ToTensor(),
                #     transforms.Normalize(mean=0, std=65535)  # 假设灰度值范围为[0, 65535]
                # ])
                # # 应用转换
                # tensor_image = transform(np.array(image).astype(float))
                # # 将张量的值归一化到[0, 1]范围
                # img_aux = tensor_image.clamp(0, 1).float()

        # label image
        label = Image.open(self.label_list[index]).convert('RGB')
        label = self.resize_label(label)
        label = np.array(label)
        label = one_hot_it(label, self.label_info).astype(np.uint8)

        if not self.aux_concat:
            img = Image.fromarray(img).convert('RGB')  # array转换为image
        img = self.to_tensor(img).float()
        # img = self.normalization(img)                  #数据归一化/标准化
        label = np.transpose(label, [2, 0, 1]).astype(np.float32)  # 将shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]
        label = torch.from_numpy(label)  # 转化为张量
        return img, label, img_aux

    def __len__(self):
        return len(self.image_list)
    ####cloudsen12 augmentation  数据增强
    # def randomRotation(image, label, mode=Image.BICUBIC):
    #
    #     """
    #      对图像进行随机任意角度(0~360度)旋转
    #     :param mode 邻近插值,双线性插值,双三次B样条插值(default)
    #     :param image PIL的图像image
    #     :return: 旋转转之后的图像
    #     """
    #     random_angle = np.random.randint(1, 360)
    #     return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)
    #
    # def randomCrop(image, label):
    #     """
    #     对图像随意剪切,考虑到图像大小范围(68,68),使用一个一个大于(36*36)的窗口进行截图
    #     :param image: PIL的图像image
    #     :return: 剪切之后的图像
    #     """
    #     image_width = image.size[0]        # 0 维处的 元素个数 ， 0维即 高度
    #     image_height = image.size[1]       # 1维处的元素个数   ， 1维即 宽度
    #
    #     crop_win_size = np.random.randint(40, 68)
    #     random_region = (
    #         (image_width - crop_win_size) >> 1, (image_height - crop_win_size) >> 1, (image_width + crop_win_size) >> 1,
    #         (image_height + crop_win_size) >> 1)
    #     return image.crop(random_region), label
    #
    # @staticmethod
    # def randomGaussian(image, label, mean=0.2, sigma=0.3):
    #     """
    #      对图像进行高斯噪声处理
    #     :param image:
    #     :return:
    #     """
    #
    #     def gaussianNoisy(im, mean=0.2, sigma=0.3):
    #         """
    #         对图像做高斯噪音处理
    #         :param im: 单通道图像
    #         :param mean: 偏移量
    #         :param sigma: 标准差
    #         :return:
    #         """
    #         for _i in range(len(im)):
    #             im[_i] += random.gauss(mean, sigma)          #im 是列表 ， im[5] 即 列表中第4个元素
    #         return im
    #     # 将图像转化成数组
    #     img = np.asarray(image)
    #     img.flags.writeable = True  # 将数组改为读写模式
    #     width, height = img.shape[:2]
    #     img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
    #     img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
    #     img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
    #     img[:, :, 0] = img_r.reshape([width, height])
    #     img[:, :, 1] = img_g.reshape([width, height])
    #     img[:, :, 2] = img_b.reshape([width, height])
    #     return Image.fromarray(np.uint8(img)), label


# class SG(Data.Dataset):
#     def __init__(self, image_path, label_path, csv_path, scale, mode="train"):
#         super().__init__()
#         self.img_list = glob.glob(os.path.join(image_path, '*.png'))
#         self.label_list = glob.glob(os.path.join(label_path, '*.png'))
#         self.label_info = get_label_info(csv_path)
#         self.colormap = [self.label_info[key] for key in self.label_info]
#
#         try:
#             assert (len(self.img_list)) == (len(self.label_list))
#         except AssertionError as AE:
#             print('Samples and labels are not match!')
#         random.seed(32)
#         random.shuffle(self.img_list)
#         random.seed(32)
#         random.shuffle(self.label_list)
#
#     def __getitem__(self, idx):
#         img = self.img_list[idx]
#         label = self.label_list[idx]
#         img = Image.open(img)
#         img = transforms.ToTensor()(img)
#
#         label = Image.open(label).convert('RGB')
#         label = np.array(label)
#         label = one_hot_it(label, self.label_info).astype(np.uint8)
#         label = np.transpose(label, [2, 0, 1]).astype(np.float32)  # 将shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]
#         label = torch.from_numpy(label)
#
#         return img, label
#
#     def __len__(self):
#         return len(self.img_list)
#
#     def img_transforms(self, cloudsen12, label, colormap):
#         cloudsen12 = transforms.ToTensor()(cloudsen12)
#         label = torch.from_numpy(image2label(label, colormap))
#         return cloudsen12, label

# class SG(Data.Dataset):
#     def __init__(self, image_path, label_path, csv_path, scale, mode="train"):
#         super().__init__()
#         self.img_list = glob.glob(os.path.join(image_path, '*.png'))
#         self.label_list = glob.glob(os.path.join(label_path, '*.png'))
#         self.label_info = get_label_info(csv_path)
#         self.colormap = [self.label_info[key] for key in self.label_info]
#
#         try:
#             assert (len(self.img_list)) == (len(self.label_list))
#         except AssertionError as AE:
#             print('Samples and labels are not match!')
#         random.seed(32)
#         random.shuffle(self.img_list)
#         random.seed(32)
#         random.shuffle(self.label_list)
#
#     def __getitem__(self, idx):
#         img = self.img_list[idx]
#         label = self.label_list[idx]
#         img = Image.open(img)
#         label = Image.open(label).convert('RGB')
#         img, label = self.img_transforms(img, label, self.colormap)
#         return img, label.long()
#
#     def __len__(self):
#         return len(self.img_list)
#
#     def img_transforms(self, cloudsen12, label, colormap):
#         cloudsen12 = transforms.ToTensor()(cloudsen12)
#         label = torch.from_numpy(image2label(label, colormap))
#         return cloudsen12, label


def get_label_info(csv_path):
    ann = pd.read_csv(csv_path)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label


def image2label(image, colormap):
    cm2lbl = np.zeros(256 ** 3)
    for i, cm in enumerate(colormap):
        cm2lbl[(cm[0] * 256 * 256 + cm[1] * 256 + cm[2])] = i
    image = np.array(image, dtype='int64')
    np.set_printoptions(threshold=np.inf)
    ix = (image[:, :, 0] * 256 * 256 + image[:, :, 1] * 256 + image[:, :, 2])
    image2 = cm2lbl[ix]
    return image2


if __name__ == '__main__':
    data = HyDataset(
        r'datasets/SPARCS/train',
        r'datasets/SPARCS/class_dict.csv',
        scale=(256, 256),
        mode='train'
    )
    print(len(data))
    for i, (image, label, aux) in enumerate(data):
        if i < 1:
            print(image.shape)
            print(label.shape)

