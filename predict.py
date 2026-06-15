import argparse
import glob
import os
import time
import warnings

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import get_label_info, reverse_one_hot, colour_code_segmentation, one_hot_it

warnings.filterwarnings('ignore')


class HyDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, csv_path, scale, mode="train", aux_bands=[]):
        super().__init__()
        self.mode = mode
        self.use_auxbands = (len(aux_bands) > 0)
        # img_root_path = os.path.join(data_root, 'img')
        img_root_path = data_root
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
        # try:
        #     assert (len(self.image_list) == len(self.label_list))
        # except AssertionError as AE:
        #     print("Sample and Label not match!")
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
            aux_img_np = []
            # k: band name v: images list of the band
            for k, v in self.image_aux_dict.items():
                # 8bit
                tmp = Image.open(v[index]).convert('L')
                tmp = self.resize_img(tmp)
                aux_img_np.append(np.array(tmp))

            aux_img_np = np.stack(aux_img_np, axis=0)
            img_aux = self.to_tensor(np.transpose(aux_img_np, (1, 2, 0))).float()

            # # 16bit
            # # 定义转换
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
        # label = Image.open(self.label_list[index]).convert('RGB')
        # label = self.resize_label(label)
        # label = np.array(label)
        # label = one_hot_it(label, self.label_info).astype(np.uint8)

        img = Image.fromarray(img).convert('RGB')  # array转换为image
        img = self.to_tensor(img).float()
        # img = self.normalization(img)                  #数据归一化/标准化
        # label = np.transpose(label, [2, 0, 1]).astype(np.float32)  # 将shape为(H,W,C)的numpy.ndarray，转换成形状为[C,H,W]
        # label = torch.from_numpy(label)  # 转化为张量
        return img, torch.Tensor(), img_aux, self.image_list[index]

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


def predict_on_image(model, args, epoch, csv_path, datetime, checkpoint_path=None):
    # pre-processing on image

    # test_path = os.path.join(args.demo_root, args.demo_name)
    test_path = os.path.join(args.demo_root, args.demo_name)
    test_list = os.listdir(test_path)
    test_list_1 = []
    for file in test_list:
        if os.path.isfile(os.path.join(test_path, file)):
            test_list_1.append(file)

    if checkpoint_path is not None:
        print('load model from %s...' % checkpoint_path)
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        print('Done!')

    test_list = test_list_1
    # save_path = os.path.join(args.demo_root, args.demo_name, args.model_name)
    save_path = os.path.join(args.demo_root, args.demo_name, args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # val_root = os.path.join(args.demo_root, args.demo_name)
    val_root = os.path.join(args.demo_root, args.demo_name)
    dataset_val = HyDataset(data_root=val_root, csv_path=csv_path, scale=(args.crop_height, args.crop_width),
                            mode='val', aux_bands=[])
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                persistent_workers=True if args.num_workers > 0 else False)
    tq = tqdm(total=len(dataloader_val))  # 进度条配置
    for i, (data, label, data_aux, _) in enumerate(dataloader_val):
        # read csv label path
        label_info = get_label_info(csv_path)
        # predict
        model.eval()
        predict = model(data.cuda())

        w = predict.size()[-1]
        c = predict.size()[-3]
        predict = predict.resize(c, w, w)
        predict = reverse_one_hot(predict)  # 此处返回的是HWC 最后一维处最大值的序号，用来判断像素点的颜色

        predict = colour_code_segmentation(np.array(predict.cpu()), label_info)  # 对每一个像素点进行分类，得到分类后的图片数据
        predict = cv2.resize(np.uint8(predict),
                             (args.crop_height, args.crop_width))  # 数据类型转换为unit8，， uint8为无符号整型数据, 范围是从0–255
        # save_name = f'{i[:-4]}_epoch_%d.png' % epoch
        index = _[0].rfind('\\') + 1
        # cv2.imwrite(os.path.join(args.demo_root, args.demo_name, args.model_name, f'{_[0][index:-4]}_predict.png'),
        #             cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_path, f'{_[0][index:-4]}_predict.png'),
                    cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
        # cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
        tq.update(1)  # 进度条刷新
    tq.close()

def predict_on_image_1(model, args, epoch, csv_path, datetime, checkpoint_path=None):
    # pre-processing on image

    # # test_path = os.path.join(args.demo_root, args.demo_name)
    # test_path = os.path.join(args.demo_root, args.demo_name)
    # test_list = os.listdir(test_path)
    # test_list_1 = []
    # for file in test_list:
    #     if os.path.isfile(os.path.join(test_path, file)):
    #         test_list_1.append(file)
    #
    if checkpoint_path is not None:
        print('load model from %s...' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)

        # 修复键名：移除 "_orig_mod."
        fixed_checkpoint = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}

        # 加载修正后的 checkpoint
        model.load_state_dict(fixed_checkpoint, strict=True)
        print('Done!')
    #
    # test_list = test_list_1
    # save_path = os.path.join(args.demo_root, args.demo_name, args.model_name)
    save_path = os.path.join(args.demo_root, args.demo_name,args.model_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    val_root = os.path.join(args.demo_root, args.demo_name)
    # val_root = os.path.join(args.demo_root, args.demo_name)
    dataset_val = HyDataset(data_root=val_root, csv_path=csv_path, scale=(args.crop_height, args.crop_width),
                            mode='val', aux_bands=[])
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=args.num_workers,
                                persistent_workers=True if args.num_workers > 0 else False)
    tq = tqdm(total=len(dataloader_val))  # 进度条配置
    for i, (data, label, data_aux, _) in enumerate(dataloader_val):
        # read csv label path
        label_info = get_label_info(csv_path)
        # predict
        model.eval()
        predict = model(data.cuda())

        w = predict.size()[-1]
        c = predict.size()[-3]
        predict = predict.resize(c, w, w)
        predict = reverse_one_hot(predict)  # 此处返回的是HWC 最后一维处最大值的序号，用来判断像素点的颜色

        predict = colour_code_segmentation(np.array(predict.cpu()), label_info)  # 对每一个像素点进行分类，得到分类后的图片数据
        predict = cv2.resize(np.uint8(predict),
                             (args.crop_height, args.crop_width))  # 数据类型转换为unit8，， uint8为无符号整型数据, 范围是从0–255
        # save_name = f'{i[:-4]}_epoch_%d.png' % epoch
        # index = _[0].rfind('\\') + 1
        # cv2.imwrite(os.path.join(args.demo_root, args.demo_name, args.model_name, f'{_[0][index:-4]}_predict.png'),
        #             cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(save_path, _[0][_[0].rfind('/') + 1:]),
                    cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
        # cv2.imwrite(save_path, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))
        tq.update(1)  # 进度条刷新
    tq.close()


# def predict1(model, demo_path, size, load_state_dict, csv_path, save_path):
#     test_list = os.listdir(demo_path)
#     for i in test_list:
#         image = cv2.imread(demo_path + i, -1)  # 读入一张图片数据，图片为BGR形式的数组
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 将BGR图片数组 转换为RGB图片数组
#
#         image = Image.fromarray(image).convert('RGB')  # 将数组转换为RGB图片
#         image = transforms.ToTensor()(image).unsqueeze(0)  # 标准化后，在第0维增加一个维度
#         # read csv label path
#         label_info = get_label_info(csv_path)
#         # predict
#         # model.eval()
#         model.load_state_dict(torch.load(load_state_dict))
#         model.eval()
#         predict = model(image)
#
#         w = predict.size()[-1]
#         c = predict.size()[-3]
#         predict = predict.resize(c, w, w)
#         predict = reverse_one_hot(predict)  # 此处返回的是HWC 最后一维处最大值的序号，用来判断像素点的颜色
#
#         predict = colour_code_segmentation(np.array(predict.cpu()), label_info)  # 对每一个像素点进行分类，得到分类后的图片数据
#         predict = cv2.resize(np.uint8(predict), (size, size))  # 数据类型转换为unit8，， uint8为无符号整型数据, 范围是从0–255
#         save_path1 = f'/{i[:-4]}.png'
#         cv2.imwrite(save_path + save_path1, cv2.cvtColor(np.uint8(predict), cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    # Example inference. Provide a trained checkpoint and a folder of demo images
    # under <demo_root>/<demo_name>/img (see dataset.py / README for the layout).
    from models.specgatenet import SpecGateNet

    parser = argparse.ArgumentParser()
    parser.add_argument('--crop_height', type=int, default=224)
    parser.add_argument('--crop_width', type=int, default=224)
    parser.add_argument('--demo_root', type=str, default='demo')
    parser.add_argument('--demo_name', type=str, default='demo')
    parser.add_argument('--model_name', type=str, default='SpecGateNet')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--csv_path', type=str, default='datasets/SPARCS/class_dict.csv')
    parser.add_argument('--checkpoint_path', type=str, default=None)
    args = parser.parse_args()

    net = SpecGateNet(size=args.crop_height, n_classes=args.num_classes,
                      cnn_pretrained=False, aux_loss=False).cuda()
    predict_on_image(net, args, epoch=-1, csv_path=args.csv_path, datetime=None,
                     checkpoint_path=args.checkpoint_path)
