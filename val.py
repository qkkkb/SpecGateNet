import argparse
import os
import time
import warnings
from functools import partial

import torch
import numpy as np
import tqdm
from predict import predict_on_image
from utils import reverse_one_hot, fast_hist, per_class_iu, coutpixel_class3, coutpixel_class4, coutpixel_class2, \
    coutpixel_class5, coutpixel_class7
from dataset import HyDataset
from torch.utils.data import DataLoader
warnings.filterwarnings('ignore')


def val_class5(args, model, dataloader, csv_path, epoch, writer, datetime, loss_train_mean):
    # 输出一张图片查看训练效果
    if args.enable_demo:
        predict_on_image(model, args, epoch, csv_path)

    start = time.time()
    with torch.no_grad():
        model.eval()
        hist = np.zeros((args.num_classes, args.num_classes))  # 返回n*n 的全0 数组

        n00 = 0
        n01 = 0
        n02 = 0
        n03 = 0
        n04 = 0

        n10 = 0
        n11 = 0
        n12 = 0
        n13 = 0
        n14 = 0

        n20 = 0
        n21 = 0
        n22 = 0
        n23 = 0
        n24 = 0

        n30 = 0
        n31 = 0
        n32 = 0
        n33 = 0
        n34 = 0

        n40 = 0
        n41 = 0
        n42 = 0
        n43 = 0
        n44 = 0

        val_loss_record = []

        tq = tqdm.tqdm(total=len(dataloader))  # 进度条配置
        tq.set_description('val progress')  # 进度条配置

        for i, (data, label, _) in enumerate(dataloader):
            data = data.cuda()
            label = label.cuda()
            out = model(data)
            # 计算val损失
            loss = torch.nn.CrossEntropyLoss()(out, label)  # loss = loss (output1,label)
            val_loss_record.append(loss.item())

            # get RGB predict image
            predict = out.squeeze()  # 降维 ， 去除维度值为1的 维度
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())
            # predict_one=predict.flatten()
            # get RGB label image
            label = label.squeeze()
            label = reverse_one_hot(label)
            label = np.array(label.cpu())
            # label_one=label.flatten()
            # compute per pixel accuracy
            tmp_hist = fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            hist += tmp_hist
            print(np.mean(per_class_iu(tmp_hist)))
            N00, N01, N02, N03, N04, N10, N11, N12, N13, N14, N20, N21, N22, N23, N24, N30, N31, N32, N33, N34, N40, N41, N42, N43, N44 = coutpixel_class5(
                label,
                predict)

            n00 = N00 + n00
            n01 = N01 + n01
            n02 = N02 + n02
            n03 = N03 + n03
            n04 = N04 + n04

            n10 = N10 + n10
            n11 = N11 + n11
            n12 = N12 + n12
            n13 = N13 + n13
            n14 = N14 + n14

            n20 = N20 + n20
            n21 = N21 + n21
            n22 = N22 + n22
            n23 = N23 + n23
            n24 = N24 + n24

            n30 = N30 + n30
            n31 = N31 + n31
            n32 = N32 + n32
            n33 = N33 + n33
            n34 = N34 + n34

            n40 = N40 + n40
            n41 = N41 + n41
            n42 = N42 + n42
            n43 = N43 + n43
            n44 = N44 + n44
            tq.update(1)  # 进度条刷新

        tq.close()
        loss_val_mean = np.mean(val_loss_record)
        sum_elements = n00 + n01 + n02 + n03 + n04 + n10 + n11 + n12 + n13 + n14 + n20 + n21 + n22 + n23 + n24 + n30 + n31 + n32 + n33 + n34 + n40 + n41 + n42 + n43 + n44
        PA = (n00 + n11 + n22 + n33 + n44) / sum_elements
        p0 = n00 / (n00 + n10 + n20 + n30 + n40)  # CPA
        p1 = n11 / (n01 + n11 + n21 + n31 + n41)
        p2 = n22 / (n02 + n12 + n22 + n32 + n42)
        p3 = n33 / (n03 + n13 + n23 + n33 + n43)
        p4 = n44 / (n04 + n14 + n24 + n34 + n44)

        r0 = n00 / (n00 + n01 + n02 + n03 + n04)
        r1 = n11 / (n10 + n11 + n12 + n13 + n14)
        r2 = n22 / (n20 + n21 + n22 + n23 + n24)
        r3 = n33 / (n30 + n31 + n32 + n33 + n34)
        r4 = n44 / (n40 + n41 + n42 + n43 + n44)
        R = np.mean((r0, r1, r2, r3, r4))

        F1_0 = (r0 + p0) / 2
        F1_1 = (r1 + p1) / 2
        F1_2 = (r2 + p2) / 2
        F1_3 = (r3 + p3) / 2
        F1_4 = (r4 + p4) / 2
        F1 = np.mean((F1_0, F1_1, F1_2, F1_3, F1_4))

        OA0 = (n00 + n11 + n12 + n13 + n21 + n22 + n23 + n31 + n32 + n33) / sum_elements
        OA1 = (n11 + n00 + n02 + n03 + n20 + n22 + n23 + n30 + n32 + n33) / sum_elements
        OA2 = (n00 + n01 + n03 + n10 + n11 + n13 + n30 + n31 + n33 + n22) / sum_elements
        OA3 = (n33 + n00 + n01 + n02 + n10 + n11 + n12 + n20 + n21 + n22) / sum_elements
        OA4 = (n33 + n00 + n01 + n02 + n10 + n11 + n12 + n20 + n21 + n22) / sum_elements

        MPA = (p0 + p1 + p2 + p3 + p4) / 5
        miou = np.mean(per_class_iu(hist))

        writer.add_scalar('{}_loss_val_mean'.format('val'), loss_val_mean, epoch)
        writer.add_scalar('{}_MIoU'.format('val'), miou, epoch)
        writer.add_scalar('{}_PA'.format('val'), PA, epoch)
        writer.add_scalar('{}_MPA'.format('val'), MPA, epoch)

        end = time.time()
        print('loss for val :{:.6f}'.format(loss_val_mean))
        print('PA  : {:.5f}'.format(PA))  # pixcal-accuracy
        print('MPA :{:.5} '.format(MPA))
        print('mIoU: {:.5f}'.format(miou))  # mean intersection over union
        print("Time:{:.3f}s".format(end - start))

        # 将验证的结果记录在result.txt以及summary对应文件下的txt文件中
        str_ = ("%15.5g;" * 28) % (
            epoch, loss_train_mean, loss_val_mean, PA, MPA, miou, R, F1, p0, p1, p2, p3, p4, r0, r1, r2, r3, r4, F1_0,
            F1_1,
            F1_2, F1_3, F1_4, OA0, OA1,
            OA2, OA3, OA4)
        with open(f'result/{datetime}_{args.model_name}.txt', 'a') as f:
            f.write(str_ + '\n')
        return miou


def val_class7(args, model, dataloader, csv_path, epoch, writer, datetime, loss_train_mean):
    # 输出一张图片查看训练效果
    # if args.enable_demo:
    # predict_on_image(model, args, epoch, csv_path, datetime)
    aux_bands = (len(args.aux_bands) > 0)
    start = time.time()
    with torch.no_grad():
        model.eval()
        hist = np.zeros((args.num_classes, args.num_classes))  # 返回n*n 的全0 数组

        n00 = 0
        n01 = 0
        n02 = 0
        n03 = 0
        n04 = 0
        n05 = 0
        n06 = 0

        n10 = 0
        n11 = 0
        n12 = 0
        n13 = 0
        n14 = 0
        n15 = 0
        n16 = 0

        n20 = 0
        n21 = 0
        n22 = 0
        n23 = 0
        n24 = 0
        n25 = 0
        n26 = 0

        n30 = 0
        n31 = 0
        n32 = 0
        n33 = 0
        n34 = 0
        n35 = 0
        n36 = 0

        n40 = 0
        n41 = 0
        n42 = 0
        n43 = 0
        n44 = 0
        n45 = 0
        n46 = 0

        n50 = 0
        n51 = 0
        n52 = 0
        n53 = 0
        n54 = 0
        n55 = 0
        n56 = 0

        n60 = 0
        n61 = 0
        n62 = 0
        n63 = 0
        n64 = 0
        n65 = 0
        n66 = 0

        val_loss_record = []

        tq = tqdm.tqdm(total=len(dataloader))  # 进度条配置
        tq.set_description('val progress')  # 进度条配置

        for i, (data, label, data_aux) in enumerate(dataloader):
            data = data.cuda()
            label = label.cuda()

            if aux_bands:
                out = model(data, data_aux.cuda())
            else:
                out = model(data)
            # 计算val损失
            loss = torch.nn.CrossEntropyLoss()(out, label)  # loss = loss (output1,label)
            val_loss_record.append(loss.item())

            # get RGB predict image
            predict = out.squeeze()  # 降维 ， 去除维度值为1的 维度
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())
            # predict_one=predict.flatten()
            # get RGB label image
            label = label.squeeze()
            label = reverse_one_hot(label)
            label = np.array(label.cpu())
            # label_one=label.flatten()
            # compute per pixel accuracy
            tmp_hist = fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            hist += tmp_hist
            # print(np.mean(per_class_iu(tmp_hist)))
            N00, N01, N02, N03, N04, N05, N06, N10, N11, N12, N13, N14, N15, N16, N20, N21, N22, N23, N24, N25, N26, N30, N31, N32, N33, N34, N35, N36, N40, N41, N42, N43, N44, N45, N46, N50, N51, N52, N53, N54, N55, N56, N60, N61, N62, N63, N64, N65, N66 = coutpixel_class7(
                label, predict)

            n00 = N00 + n00
            n01 = N01 + n01
            n02 = N02 + n02
            n03 = N03 + n03
            n04 = N04 + n04
            n05 = N05 + n05
            n06 = N06 + n06

            n10 = N10 + n10
            n11 = N11 + n11
            n12 = N12 + n12
            n13 = N13 + n13
            n14 = N14 + n14
            n15 = N15 + n15
            n16 = N16 + n16

            n20 = N20 + n20
            n21 = N21 + n21
            n22 = N22 + n22
            n23 = N23 + n23
            n24 = N24 + n24
            n25 = N25 + n25
            n26 = N26 + n26

            n30 = N30 + n30
            n31 = N31 + n31
            n32 = N32 + n32
            n33 = N33 + n33
            n34 = N34 + n34
            n35 = N35 + n35
            n36 = N36 + n36

            n40 = N40 + n40
            n41 = N41 + n41
            n42 = N42 + n42
            n43 = N43 + n43
            n44 = N44 + n44
            n45 = N45 + n45
            n46 = N46 + n46

            n50 = N50 + n50
            n51 = N51 + n51
            n52 = N52 + n52
            n53 = N53 + n53
            n54 = N54 + n54
            n55 = N55 + n55
            n56 = N56 + n56

            n60 = N60 + n60
            n61 = N61 + n61
            n62 = N62 + n62
            n63 = N63 + n63
            n64 = N64 + n64
            n65 = N65 + n65
            n66 = N66 + n66

            tq.update(1)  # 进度条刷新

        tq.close()
        loss_val_mean = np.mean(val_loss_record)
        sum_elements = n00 + n01 + n02 + n03 + n04 + n05 + n06 + n10 + n11 + n12 + n13 + n14 + n15 + n16 + n20 + n21 + n22 + n23 + n24 + n25 + n26 + n30 + n31 + n32 + n33 + n34 + n35 + n36 + n40 + n41 + n42 + n43 + n44 + n45 + n46 + n50 + n51 + n52 + n53 + n54 + n55 + n56 + n60 + n61 + n62 + n63 + n64 + n65 + n66
        PA = (n00 + n11 + n22 + n33 + n44 + n55 + n66) / sum_elements
        p0 = n00 / (n00 + n10 + n20 + n30 + n40 + n50 + n60)  # CPA
        p1 = n11 / (n01 + n11 + n21 + n31 + n41 + n51 + n61)
        p2 = n22 / (n02 + n12 + n22 + n32 + n42 + n52 + n62)
        p3 = n33 / (n03 + n13 + n23 + n33 + n43 + n53 + n63)
        p4 = n44 / (n04 + n14 + n24 + n34 + n44 + n54 + n64)
        p5 = n55 / (n05 + n15 + n25 + n35 + n45 + n55 + n65)
        p6 = n66 / (n06 + n16 + n26 + n36 + n46 + n56 + n66)

        r0 = n00 / (n00 + n01 + n02 + n03 + n04 + n05 + n06)
        r1 = n11 / (n10 + n11 + n12 + n13 + n14 + n15 + n16)
        r2 = n22 / (n20 + n21 + n22 + n23 + n24 + n25 + n26)
        r3 = n33 / (n30 + n31 + n32 + n33 + n34 + n35 + n36)
        r4 = n44 / (n40 + n41 + n42 + n43 + n44 + n45 + n46)
        r5 = n55 / (n50 + n51 + n52 + n53 + n54 + n55 + n56)
        r6 = n66 / (n60 + n61 + n62 + n63 + n64 + n65 + n66)

        R = np.mean((r0, r1, r2, r3, r4, r5, r6))

        F1_0 = 2 * r0 * p0 / (r0 + p0)
        F1_1 = 2 * r1 * p1 / (r1 + p1)
        F1_2 = 2 * r2 * p2 / (r2 + p2)
        F1_3 = 2 * r3 * p3 / (r3 + p3)
        F1_4 = 2 * r4 * p4 / (r4 + p4)
        F1_5 = 2 * r5 * p5 / (r5 + p5)
        F1_6 = 2 * r6 * p6 / (r6 + p6)

        F1 = np.mean((F1_0, F1_1, F1_2, F1_3, F1_4, F1_5, F1_6))

        # OA0 = (n00 + n11 + n12 + n13 + n21 + n22 + n23 + n31 + n32 + n33) / sum_elements
        # OA1 = (n11 + n00 + n02 + n03 + n20 + n22 + n23 + n30 + n32 + n33) / sum_elements
        # OA2 = (n00 + n01 + n03 + n10 + n11 + n13 + n30 + n31 + n33 + n22) / sum_elements
        # OA3 = (n33 + n00 + n01 + n02 + n10 + n11 + n12 + n20 + n21 + n22) / sum_elements
        # OA4 = (n33 + n00 + n01 + n02 + n10 + n11 + n12 + n20 + n21 + n22) / sum_elements

        MPA = (p0 + p1 + p2 + p3 + p4 + p5 + p6) / 7
        miou = np.mean(per_class_iu(hist))

        writer.add_scalar('{}_loss_val_mean'.format('val'), loss_val_mean, epoch)
        writer.add_scalar('{}_MIoU'.format('val'), miou, epoch)
        writer.add_scalar('{}_PA'.format('val'), PA, epoch)
        writer.add_scalar('{}_MPA'.format('val'), MPA, epoch)

        end = time.time()
        print('loss for val :{:.6f}'.format(loss_val_mean))
        print('PA  : {:.5f}'.format(PA))  # pixcal-accuracy
        print('MPA :{:.5} '.format(MPA))
        print('mIoU: {:.5f}'.format(miou))  # mean intersection over union
        print("Time:{:.3f}s".format(end - start))

        # 将验证的结果记录在result.txt以及summary对应文件下的txt文件中
        str_ = ("%15.5g;" * 29) % (
            epoch, loss_train_mean, loss_val_mean, PA, MPA, miou, R, F1, p0, p1, p2, p3, p4, p5, p6, r0, r1, r2, r3, r4,
            r5, r6, F1_0,
            F1_1,
            F1_2, F1_3, F1_4, F1_5, F1_6)
        with open(f'result/{datetime}_{args.model_name}.txt', 'a') as f:
            f.write(str_ + '\n')
        return miou


def val_class4(args, model, dataloader, csv_path, epoch, writer, datetime, loss_train_mean):
    # 输出一张图片查看训练效果
    if args.enable_demo:
        predict_on_image(model, args, epoch, csv_path)

    start = time.time()
    with torch.no_grad():
        model.eval()
        hist = np.zeros((args.num_classes, args.num_classes))  # 返回n*n 的全0 数组

        n00 = 0
        n01 = 0
        n02 = 0
        n03 = 0

        n10 = 0
        n11 = 0
        n12 = 0
        n13 = 0

        n20 = 0
        n21 = 0
        n22 = 0
        n23 = 0

        n30 = 0
        n31 = 0
        n32 = 0
        n33 = 0

        val_loss_record = []
        for i, (data, label) in enumerate(dataloader):
            data = data.cuda()
            label = label.cuda()

            out = model(data)
            # 计算val损失
            loss = torch.nn.BCEWithLogitsLoss()(out, label)  # loss = loss (output1,label)
            val_loss_record.append(loss.item())

            # get RGB predict image
            predict = out.squeeze()  # 降维 ， 去除维度值为1的 维度
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())
            # predict_one=predict.flatten()
            # get RGB label image
            label = label.squeeze()
            label = reverse_one_hot(label)
            label = np.array(label.cpu())
            # label_one=label.flatten()
            # compute per pixel accuracy
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            N00, N01, N02, N03, N10, N11, N12, N13, N20, N21, N22, N23, N30, N31, N32, N33 = coutpixel_class4(label,
                                                                                                              predict)

            n00 = N00 + n00
            n01 = N01 + n01
            n02 = N02 + n02
            n03 = N03 + n03

            n10 = N10 + n10
            n11 = N11 + n11
            n12 = N12 + n12
            n13 = N13 + n13

            n20 = N20 + n20
            n21 = N21 + n21
            n22 = N22 + n22
            n23 = N23 + n23

            n30 = N30 + n30
            n31 = N31 + n31
            n32 = N32 + n32
            n33 = N33 + n33

        loss_val_mean = np.mean(val_loss_record)
        sum_elements = n00 + n01 + n02 + n03 + n10 + n11 + n12 + n13 + n20 + n21 + n22 + n23 + n30 + n31 + n32 + n33
        PA = (n00 + n11 + n22 + n33) / sum_elements
        p0 = n00 / (n00 + n10 + n20 + n30)  # CPA
        p1 = n11 / (n01 + n11 + n21 + n31)
        p2 = n22 / (n02 + n12 + n22 + n32)
        p3 = n33 / (n03 + n13 + n23 + n33)

        R1 = n00 / (n00 + n01 + n02 + n03)
        R2 = n11 / (n10 + n11 + n12 + n13)
        R3 = n22 / (n20 + n21 + n22 + n23)
        R4 = n33 / (n30 + n31 + n32 + n33)

        F1_1 = (R1 + p0) / 2
        F1_2 = (R2 + p1) / 2
        F1_3 = (R3 + p2) / 2
        F1_4 = (R4 + p3) / 2

        OA1 = (n00 + n11 + n12 + n13 + n21 + n22 + n23 + n31 + n32 + n33) / sum_elements
        OA2 = (n11 + n00 + n02 + n03 + n20 + n22 + n23 + n30 + n32 + n33) / sum_elements
        OA3 = (n00 + n01 + n03 + n10 + n11 + n13 + n30 + n31 + n33 + n22) / sum_elements
        OA4 = (n33 + n00 + n01 + n02 + n10 + n11 + n12 + n20 + n21 + n22) / sum_elements

        MPA = (p0 + p1 + p2 + p3) / 4
        miou = np.mean(per_class_iu(hist))
        end = time.time()

        writer.add_scalar('{}_loss_val_mean'.format('val'), loss_val_mean, epoch)
        writer.add_scalar('{}_MIoU'.format('val'), miou, epoch)
        writer.add_scalar('{}_PA'.format('val'), PA, epoch)
        writer.add_scalar('{}_MPA'.format('val'), MPA, epoch)

        print('loss for val :{:.6f}'.format(loss_val_mean))
        print('PA  : {:.5f}'.format(PA))  # pixcal-accuracy
        print('MPA :{:.5} '.format(MPA))
        print('mIoU: {:.5f}'.format(miou))  # mean intersection over union
        print("Time:{:.3f}s".format(end - start))

        # 将验证的结果记录在result.txt以及summary对应文件下的txt文件中
        str_ = ("%15.5g;" * 6) % (epoch, loss_train_mean, loss_val_mean, PA, MPA, miou)
        with open(f'result/{datetime}_{args.model_name}.txt', 'a') as f:
            f.write(str_ + '\n')
        return miou


# def val_class3(args, model, dataloader, csv_path, epoch, loss_train_mean, writer, datetime):
#     # 输出一张图片查看训练效果
#     if args.enable_demo:
#         predict_on_image(model, args, epoch, csv_path)
#     aux_bands = (len(args.aux_bands) > 0)
#     start = time.time()
#     with torch.inference_mode():
#         model.eval()
#         hist = np.zeros((args.num_classes, args.num_classes))  # 返回n*n 的全0 数组

#         n00 = 0
#         n01 = 0
#         n02 = 0

#         n10 = 0
#         n11 = 0
#         n12 = 0

#         n20 = 0
#         n21 = 0
#         n22 = 0

#         val_loss_record = []

#         tq = tqdm.tqdm(total=len(dataloader))  # 进度条配置
#         tq.set_description('val progress')  # 进度条配置

#         for i, (data, label, data_aux) in enumerate(dataloader):
#             data = data.cuda()
#             label = label.cuda()

#             if aux_bands:
#                 outputs = model(data, data_aux.cuda())
#             else:
#                 outputs = model(data)

#             # 计算val损失
#             # loss = torch.nn.BCEWithLogitsLoss()(outputs, label)
#             loss = torch.nn.CrossEntropyLoss()(outputs, label)
#             val_loss_record.append(loss.item())

#             # get RGB predict image
#             predict = outputs.squeeze()  # 降维 ， 去除维度值为1的 维度
#             predict = reverse_one_hot(predict)
#             predict = np.array(predict.cpu())
#             # predict_one=predict.flatten()
#             # get RGB label image
#             label = label.squeeze()
#             label = reverse_one_hot(label)
#             label = np.array(label.cpu())
#             # label_one=label.flatten()
#             # compute per pixel accuracy
#             hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
#             N00, N01, N02, N10, N11, N12, N20, N21, N22, = coutpixel_class3(label, predict)

#             n00 = N00 + n00
#             n01 = N01 + n01
#             n02 = N02 + n02

#             n10 = N10 + n10
#             n11 = N11 + n11
#             n12 = N12 + n12

#             n20 = N20 + n20
#             n21 = N21 + n21
#             n22 = N22 + n22
#             tq.update(1)  # 进度条刷新

#         tq.close()
#         loss_val_mean = np.mean(val_loss_record)
#         sum_elements = n00 + n01 + n02 + n10 + n11 + n12 + n20 + n21 + n22

#         # 像素准确率 PA
#         # Accuracy = (TP + TN) / (TP + TN + FP + FN)
#         # 意义：对角线计算。预测结果中正确的占总预测值的比例（对角线元素值的和 / 总元素值的和）
#         PA = (n00 + n11 + n22) / sum_elements

#         # 类别像素准确率 CPA
#         # 公式：Precision = TP / (TP + FP) 或 TN / (TN + FN)
#         # 意义：竖着计算。预测结果中，某类别预测正确的概率
#         p0 = n00 / (n00 + n10 + n20)  # CPA    云
#         p1 = n11 / (n01 + n11 + n21)  # 影
#         p2 = n22 / (n02 + n12 + n22)  # void
#         # 类别平均像素准确率 MPA
#         MPA = (p0 + p1 + p2) / 3
#         miou = np.mean(per_class_iu(hist))

#         # 召回率（Recall）
#         # Recall = TP / (TP + FN) 或 TN / (TN + FP)
#         # 意义：横着计算。真实值中，某类别被预测正确的概率
#         r0 = n00 / (n00 + n01 + n02)
#         r1 = n11 / (n10 + n11 + n12)
#         r2 = n22 / (n20 + n21 + n22)

#         F1_0 = (r0 + p0) / 2
#         F1_1 = (r1 + p1) / 2
#         F1_2 = (r2 + p2) / 2

#         OA0 = (n00 + n11 + n12 + n21 + n22) / sum_elements
#         OA1 = (n11 + n00 + n02 + n20 + n22) / sum_elements
#         OA2 = (n00 + n01 + n10 + n11 + n22) / sum_elements

#         writer.add_scalar('{}_loss_val_mean'.format('val'), loss_val_mean, epoch)
#         writer.add_scalar('{}_MIoU'.format('val'), miou, epoch)
#         writer.add_scalar('{}_PA'.format('val'), PA, epoch)
#         writer.add_scalar('{}_MPA'.format('val'), MPA, epoch)

#         if args.use_wandb:
#             import wandb
#             wandb.log({"val_loss": loss_val_mean,
#                        "val_MIoU": miou,
#                        "val_PA": PA,
#                        "val_MPA": MPA,
#                        "val_p0": p0,
#                        "val_p1": p1,
#                        "val_p2": p2,
#                        "val_r0": r0,
#                        "val_r1": r1,
#                        "val_r2": r2,
#                        "val_F1_0": F1_0,
#                        "val_F1_1": F1_1,
#                        "val_F1_2": F1_2,
#                        "val_OA0": OA0,
#                        "val_OA1": OA1,
#                        "val_OA2": OA2,
#                        })
#         end = time.time()
#         print('loss for val :{:.6f}'.format(loss_val_mean))
#         print('PA  : {:.5f}'.format(PA))  # pixcal-accuracy
#         print('MPA :{:.5} '.format(MPA))
#         print('mIoU: {:.5f}'.format(miou))  # mean intersection over union
#         print("Time:{:.3f}s".format(end - start))

#         # 将验证的结果记录在result.txt以及summary对应文件下的txt文件中
#         str_ = ("%15.5g;" * 18) % (
#             epoch, loss_train_mean, loss_val_mean, PA, MPA, miou, p0, p1, p2, r0, r1, r2, F1_0, F1_1, F1_2, OA0, OA1,
#             OA2)
#         with open(f'result/{datetime}_{args.model_name}.txt', 'a') as f:
#             f.write(str_ + '\n')
#         return miou

def val_class3(args, model, dataloader, csv_path, epoch, writer, datetime, loss_train_mean):
    # 输出一张图片查看训练效果
    if args.enable_demo:
        predict_on_image(model, args, epoch, csv_path)
    aux_bands = (len(args.aux_bands) > 0)
    start = time.time()
    with torch.inference_mode():
        model.eval()
        hist = torch.zeros((args.num_classes, args.num_classes), device='cuda')

        # 在GPU上累积计数
        counts = {
            'n00': torch.tensor(0).cuda(),
            'n01': torch.tensor(0).cuda(),
            'n02': torch.tensor(0).cuda(),
            'n10': torch.tensor(0).cuda(),
            'n11': torch.tensor(0).cuda(),
            'n12': torch.tensor(0).cuda(),
            'n20': torch.tensor(0).cuda(),
            'n21': torch.tensor(0).cuda(),
            'n22': torch.tensor(0).cuda(),
        }

        val_loss_record = []

        tq = tqdm.tqdm(total=len(dataloader.dataset))  # 进度条配置
        tq.set_description('val progress')  # 进度条配置

        for i, (data, label, data_aux) in enumerate(dataloader):
            data = data.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            batch_size_cur = data.size(0)

            if aux_bands:
                outputs = model(data, data_aux.cuda(non_blocking=True))
            else:
                outputs = model(data)

            # 计算val损失
            # loss = torch.nn.BCEWithLogitsLoss()(outputs, label)
            target = label.argmax(dim=1).long() if label.ndim == 4 else label.long()
            loss = torch.nn.CrossEntropyLoss()(outputs, target)
            val_loss_record.append(loss.item())

            # 直接在GPU上计算预测结果
            predict = outputs.argmax(dim=1)  # 避免使用reverse_one_hot
            label = target

             # 在GPU上计算混淆矩阵元素
            counts['n00'] += ((predict == 0) & (label == 0)).sum()
            counts['n01'] += ((predict == 0) & (label == 1)).sum()
            counts['n02'] += ((predict == 0) & (label == 2)).sum()
            counts['n10'] += ((predict == 1) & (label == 0)).sum()
            counts['n11'] += ((predict == 1) & (label == 1)).sum()
            counts['n12'] += ((predict == 1) & (label == 2)).sum()
            counts['n20'] += ((predict == 2) & (label == 0)).sum()
            counts['n21'] += ((predict == 2) & (label == 1)).sum()
            counts['n22'] += ((predict == 2) & (label == 2)).sum()

            k = (label >= 0) & (label < args.num_classes)
            hist += torch.bincount(
                args.num_classes * label[k] + predict[k],
                minlength=args.num_classes ** 2
            ).reshape(args.num_classes, args.num_classes)

            tq.update(batch_size_cur)  # 进度条刷新

        tq.close()
        loss_val_mean = np.mean(val_loss_record)
        hist = hist.cpu().numpy()
        
        # 将counts移到CPU进行最终计算
        sum_elements = sum(count.cpu().item() for count in counts.values())
       

        # 像素准确率 PA
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        # 意义：对角线计算。预测结果中正确的占总预测值的比例（对角线元素值的和 / 总元素值的和）
        PA = (counts['n00'] + counts['n11'] + counts['n22']).cpu().item() / sum_elements
        
        # 类别像素准确率 CPA
        # 公式：Precision = TP / (TP + FP) 或 TN / (TN + FN)
        # 意义：竖着计算。预测结果中，某类别预测正确的概率
        p0 = counts['n00'].cpu().item() / (counts['n00'] + counts['n10'] + counts['n20']).cpu().item()
        p1 = counts['n11'].cpu().item() / (counts['n01'] + counts['n11'] + counts['n21']).cpu().item()
        p2 = counts['n22'].cpu().item() / (counts['n02'] + counts['n12'] + counts['n22']).cpu().item()
        
        # 类别平均像素准确率 MPA
        MPA = (p0 + p1 + p2) / 3
        miou = np.mean(per_class_iu(hist))

        # 召回率（Recall）
        # Recall = TP / (TP + FN) 或 TN / (TN + FP)
        # 意义：横着计算。真实值中，某类别被预测正确的概率
        r0 = counts['n00'].cpu().item() / (counts['n00'].cpu().item() + counts['n01'].cpu().item() + counts['n02'].cpu().item())
        r1 = counts['n11'].cpu().item() / (counts['n10'].cpu().item() + counts['n11'].cpu().item() + counts['n12'].cpu().item())
        r2 = counts['n22'].cpu().item() / (counts['n20'].cpu().item() + counts['n21'].cpu().item() + counts['n22'].cpu().item())

        F1_0 = (r0 + p0) / 2
        F1_1 = (r1 + p1) / 2
        F1_2 = (r2 + p2) / 2

        OA0 = (counts['n00'].cpu().item() + counts['n11'].cpu().item() + counts['n12'].cpu().item() + counts['n21'].cpu().item() + counts['n22'].cpu().item()) / sum_elements
        OA1 = (counts['n11'].cpu().item() + counts['n00'].cpu().item() + counts['n02'].cpu().item() + counts['n20'].cpu().item() + counts['n22'].cpu().item()) / sum_elements
        OA2 = (counts['n00'].cpu().item() + counts['n01'].cpu().item() + counts['n10'].cpu().item() + counts['n11'].cpu().item() + counts['n22'].cpu().item()) / sum_elements

        writer.add_scalar('{}_loss_val_mean'.format('val'), loss_val_mean, epoch)
        writer.add_scalar('{}_MIoU'.format('val'), miou, epoch)
        writer.add_scalar('{}_PA'.format('val'), PA, epoch)
        writer.add_scalar('{}_MPA'.format('val'), MPA, epoch)

        if args.use_wandb:
            import wandb
            wandb.log({"val_loss": loss_val_mean,
                       "val_MIoU": miou,
                       "val_PA": PA,
                       "val_MPA": MPA,
                       "val_p0": p0,
                       "val_p1": p1,
                       "val_p2": p2,
                       "val_r0": r0,
                       "val_r1": r1,
                       "val_r2": r2,
                       "val_F1_0": F1_0,
                       "val_F1_1": F1_1,
                       "val_F1_2": F1_2,
                       "val_OA0": OA0,
                       "val_OA1": OA1,
                       "val_OA2": OA2,
                       })
        end = time.time()
        print('loss for val :{:.6f}'.format(loss_val_mean))
        print('PA  : {:.5f}'.format(PA))  # pixcal-accuracy
        print('MPA :{:.5} '.format(MPA))
        print('mIoU: {:.5f}'.format(miou))  # mean intersection over union
        print("Time:{:.3f}s".format(end - start))

        # 将验证的结果记录在result.txt以及summary对应文件下的txt文件中
        str_ = ("%15.5g;" * 18) % (
            epoch, loss_train_mean, loss_val_mean, PA, MPA, miou, p0, p1, p2, r0, r1, r2, F1_0, F1_1, F1_2, OA0, OA1,
            OA2)
        with open(f'result/{datetime}_{args.model_name}.txt', 'a') as f:
            f.write(str_ + '\n')
        return miou


def val_class2(args, model, dataloader, csv_path, epoch, writer, datetime, loss_train_mean=0):
    # 输出一张图片查看训练效果
    if args.enable_demo:
        predict_on_image(model, args, epoch, csv_path)
    aux_bands = (len(args.aux_bands) > 0)
    start = time.time()
    with torch.no_grad():
        model.eval()
        hist = np.zeros((args.num_classes, args.num_classes))  # 返回n*n 的全0 数组

        n00 = 0
        n01 = 0

        n10 = 0
        n11 = 0

        val_loss_record = []

        tq = tqdm.tqdm(total=len(dataloader))  # 进度条配置
        tq.set_description('val progress')  # 进度条配置

        for i, (data, label, data_aux) in enumerate(dataloader):
            data = data.cuda()
            label = label.cuda()

            if aux_bands:
                outputs = model(data, data_aux.cuda())
            else:
                outputs = model(data)

            # 计算val损失
            loss = torch.nn.BCEWithLogitsLoss()(outputs, label)  # loss = loss (output1,label)
            val_loss_record.append(loss.item())

            # get RGB predict image
            predict = outputs.squeeze()  # 降维 ， 去除维度值为1的 维度
            predict = reverse_one_hot(predict)
            predict = np.array(predict.cpu())
            # predict_one=predict.flatten()
            # get RGB label image
            label = label.squeeze()
            label = reverse_one_hot(label)
            label = np.array(label.cpu())
            # label_one=label.flatten()
            # compute per pixel accuracy
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            N00, N01, N10, N11 = coutpixel_class2(label, predict)

            n00 = N00 + n00
            n01 = N01 + n01

            n10 = N10 + n10
            n11 = N11 + n11
            tq.update(1)  # 进度条刷新

        tq.close()
        loss_val_mean = np.mean(val_loss_record)
        sum_elements = n00 + n01 + n10 + n11
        PA = (n00 + n11) / sum_elements
        p0 = n00 / (n00 + n10)  # CPA
        p1 = n11 / (n01 + n11)
        MPA = (p0 + p1) / 2
        miou = np.mean(per_class_iu(hist))

        # 召回率（Recall）
        # Recall = TP / (TP + FN) 或 TN / (TN + FP)
        # 意义：横着计算。真实值中，某类别被预测正确的概率
        r0 = n00 / (n00 + n01)
        r1 = n11 / (n10 + n11)

        F1_0 = (r0 + p0) / 2
        F1_1 = (r1 + p1) / 2

        end = time.time()

        writer.add_scalar('{}_loss_val_mean'.format('val'), loss_val_mean, epoch)
        writer.add_scalar('{}_MIoU'.format('val'), miou, epoch)
        writer.add_scalar('{}_PA'.format('val'), PA, epoch)
        writer.add_scalar('{}_MPA'.format('val'), MPA, epoch)

        print('loss for val :{:.6f}'.format(loss_val_mean))
        print('PA  : {:.5f}'.format(PA))  # pixcal-accuracy
        print('MPA :{:.5} '.format(MPA))
        print('mIoU: {:.5f}'.format(miou))  # mean intersection over union
        print("Time:{:.3f}s".format(end - start))

        # 将验证的结果记录在result.txt以及summary对应文件下的txt文件中
        str_ = ("%15.5g;" * 12) % (
            epoch, loss_train_mean, loss_val_mean, PA, MPA, miou, p0, p1, r0, r1, F1_0, F1_1)
        with open(f'result/{datetime}_{args.model_name}.txt', 'a') as f:
            f.write(str_ + '\n')
        return miou


# def val_class151(args, model, dataloader, csv_path, epoch, loss_train_mean, writer, datetime):
#     if args.enable_demo:
#         predict_on_image(model, args, epoch, csv_path)
#
#     aux_bands = (len(args.aux_bands) > 0)
#     start = time.time()
#
#     with torch.no_grad():
#         model.eval()
#         hist = np.zeros((151, 151))
#
#         n = [0] * 151
#
#         val_loss_record = []
#
#         tq = tqdm.tqdm(total=len(dataloader))
#         tq.set_description('val progress')
#
#         for i, (data, label, data_aux) in enumerate(dataloader):
#             data = data.cuda()
#             label = label.cuda()
#
#             if aux_bands:
#                 outputs = model(data, data_aux.cuda())
#             else:
#                 outputs = model(data)
#
#             loss = torch.nn.CrossEntropyLoss()(outputs, label)
#             val_loss_record.append(loss.item())
#
#             predict = outputs.squeeze()
#             predict = reverse_one_hot(predict)
#             predict = np.array(predict.cpu())
#
#             label = label.squeeze()
#             label = reverse_one_hot(label)
#             label = np.array(label.cpu())
#
#             hist += fast_hist(label.flatten(), predict.flatten(), 151)
#
#             for c in range(151):
#                 n[c] += np.sum(np.logical_and(predict == c, label == c))
#
#             tq.update(1)
#
#         tq.close()
#         loss_val_mean = np.mean(val_loss_record)
#         sum_elements = np.sum(n)
#
#         PA = np.sum(n) / sum_elements
#
#         CPA = []
#         MPA = 0
#         for c in range(151):
#             p = n[c] / np.sum(predict == c)
#             CPA.append(p)
#             MPA += p
#
#         MPA /= 151
#
#         miou = np.mean(per_class_iu(hist))
#
#         writer.add_scalar('{}_loss_val_mean'.format('val'), loss_val_mean, epoch)
#         writer.add_scalar('{}_MIoU'.format('val'), miou, epoch)
#         writer.add_scalar('{}_PA'.format('val'), PA, epoch)
#         writer.add_scalar('{}_MPA'.format('val'), MPA, epoch)
#
#         if args.use_wandb:
#             import wandb
#             wandb.log({
#                 "val_loss": loss_val_mean,
#                 "val_MIoU": miou,
#                 "val_PA": PA,
#                 "val_MPA": MPA
#             })
#
#         end = time.time()
#
#         print('Loss for val: {:.6f}'.format(loss_val_mean))
#         print('PA: {:.5f}'.format(PA))
#         print('MPA: {:.5f}'.format(MPA))
#         print('mIoU: {:.5f}'.format(miou))
#         print("Time: {:.3f}s".format(end - start))
#
#         str_ = ("%15.5g;" * 6) % (epoch, loss_train_mean, loss_val_mean, PA, MPA, miou)
#
#         with open(f'result/{datetime}_{args.model_name}.txt', 'a') as f:
#             f.write(str_ + '\n')
#
#         return miou
