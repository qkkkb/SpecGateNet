import math
import os.path
from typing import Iterable, Set, List
#loss.backward()outputs = model(data)
import torch
import warnings

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
from torch import nn, einsum, Tensor
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt as distance
from utils import poly_lr_scheduler
from val import val_class2, val_class4, val_class3, val_class5, val_class7
import numpy as np
import time
import tqdm  # 进度条配置

warnings.filterwarnings('ignore')

# ★ 加速配置：cudnn benchmark（固定输入尺寸时必开）
torch.backends.cudnn.benchmark = True


class Aux_DynamicWeightedLoss(nn.Module):
    def __init__(self, num=4):
        super(Aux_DynamicWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, outputs, label):
        loss_sum = 0
        losses = []

        # calc losses
        for i in range(len(outputs)):
            losses.append(torch.nn.CrossEntropyLoss()(outputs[i], label))

        for i, loss in enumerate(losses):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class Aux_DynamicWeightedLossV2(nn.Module):
    def __init__(self, num=4):
        super(Aux_DynamicWeightedLossV2, self).__init__()

    def forward(self, outputs, label):
        loss_sum = loss_lovasz(outputs[0], label) \
                   + 0.5 * torch.nn.CrossEntropyLoss()(outputs[1], label) \
                   + 0.3 * loss_dice(outputs[2], label) \
                   + 0.2 * torch.nn.CrossEntropyLoss()(outputs[3], label)
        return loss_sum


def loss_dice(inputs, targets):
    num = targets.size(0)
    inputs = torch.sigmoid(inputs)
    flat_inputs = inputs.view(num, -1)
    flat_targets = targets.view(num, -1)
    intersection = torch.sum(flat_inputs * flat_targets, dim=1)
    union = torch.sum(flat_inputs, dim=1) + torch.sum(flat_targets, dim=1)
    dice_scores = 2 * intersection / (union + 1e-8)
    return 1 - dice_scores.mean()


def loss_lovasz(inputs, labels):
    """
    计算Lovasz Hinge Loss
    :param inputs: 预测结果，shape为[N, H, W, C]
    :param labels: 真实标签，shape为[N, H, W, C]
    :return: Lovasz Hinge Loss
    """

    def _lovasz_grad(gt_sorted):
        """
        计算梯度
        :param gt_sorted: 排序后的真实标签，shape为[N]
        :return: 梯度，shape为[N]
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    inputs = inputs.permute(0, 2, 3, 1).contiguous()
    labels = labels.permute(0, 2, 3, 1).contiguous()
    # 将logits和labels展平，方便计算
    inputs = inputs.view(-1)
    labels = labels.view(-1)

    # 样本数为0，直接返回0
    if inputs.numel() == 0:
        return inputs * 0.0

    # 计算误差
    errors = (1 - labels.float() * 2) * inputs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(nn.functional.relu(errors_sorted), grad)

    return loss


# class BoundaryLoss(nn.Module):
#     def __init__(self, beta=1):
#         super(BoundaryLoss, self).__init__()
#         self.beta = beta
#
#     def forward(self, y_pred, y_true):
#         eps = 1e-6 # 防止除零错误的小常量
#
#         # 计算边界损失
#         diff_y_pred_h = torch.pow(torch.abs(y_pred[:, :, :-1] - y_pred[:, :, 1:]), self.beta)
#         diff_y_true_h = torch.pow(torch.abs(y_true[:, :, :-1] - y_true[:, :, 1:]), self.beta)
#         diff_y_pred_w = torch.pow(torch.abs(y_pred[:, :-1, :] - y_pred[:, 1:, :]), self.beta)
#         diff_y_true_w = torch.pow(torch.abs(y_true[:, :-1, :] - y_true[:, 1:, :]), self.beta)
#         loss_h = torch.sum(diff_y_pred_h * diff_y_true_h)
#         loss_w = torch.sum(diff_y_pred_w * diff_y_true_w)
#         loss = (loss_h + loss_w) / (y_pred.shape[0] * (y_pred.shape[1] - 1 + eps) * (y_pred.shape[2] - 1 + eps))
#
#         return loss
# 根据epoch动态非线性调整Dice和Focal loss的权重，并返回最终loss
class DiceFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1, lambda1=0.1, lambda2=0.1):
        super(DiceFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def forward(self, input, target, epoch):
        smooth = 1.0
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        intersection = (input_flat * target_flat).sum()
        dice_score = (2.0 * intersection + smooth) / (input_flat.sum() + target_flat.sum() + smooth)
        dice_loss = 1.0 - dice_score

        ce_loss = nn.CrossEntropyLoss(reduction='none')(input, target)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # Calculate the dynamic weights of Dice Loss and Focal Loss
        weight1 = math.exp(-self.lambda1 * epoch)
        weight2 = math.exp(self.lambda2 * epoch)
        # Combine Dice Loss and Focal Loss
        loss = weight1 * dice_loss + weight2 * focal_loss.mean()
        return loss


# 根据epoch动态线性调整两个辅助损失函数的权重，返回权重值
def weight_schedule(epoch):
    start_dice_weight = 0.9
    start_bound_weight = 0.1
    if epoch < 5:
        return start_dice_weight, start_bound_weight
    else:
        dice_weight = max(start_dice_weight - (epoch - 5) * 0.1, 0)
        bound_weight = min(start_bound_weight + (epoch - 5) * 0.1, 1)
        return dice_weight, bound_weight


def train(args, model, dataloader_train, dataloader_val, csv_path, datetime, aux_loss=False):
    step = 0
    mIoU_cache = 0.5
    writer = SummaryWriter()
    aux_bands = (len(args.aux_bands) > 0)
    # 打印表头
    s = ("%15s;" * 18) % (
        "epoch", "loss for train", "loss for val", "PA ", "MPA", "miou", 'p0', 'p1', 'p2', 'R1', 'R2', 'R3', 'F1_1',
        'F1_2',
        'F1_3', 'OA1', 'OA2', 'OA3')
    with open(f'result/{datetime}_{args.model_name}.txt', 'a') as f:
        f.write(s + '\n')

    if aux_loss:
        criterion = Aux_DynamicWeightedLossV2(num=4)
        optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=1e-4)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), args.learning_rate, weight_decay=1e-4)

    diceFocalLoss = DiceFocalLoss()

    # ★ 混合精度训练：创建 GradScaler
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(args.num_epochs):
        model.train()
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)  # 学习率自动变化
        epoch = epoch + args.epoch_start_i
        epoch = epoch + 1
        tq = tqdm.tqdm(total=len(dataloader_train.dataset))  # 进度条配置
        tq.set_description('epoch %d, lr %f' % (epoch, lr))  # 进度条配置
        loss_record = []
        dice_weight, bound_weight = weight_schedule(epoch)
        for i, (data, label, data_aux) in enumerate(dataloader_train):
            data = data.cuda(non_blocking=True)       # ★ non_blocking 配合 pin_memory
            label = label.cuda(non_blocking=True)
            batch_size_cur = data.size(0)

            # ★ 混合精度前向：autocast 自动选择 fp16/fp32
            with torch.cuda.amp.autocast():
                if aux_bands:
                    outputs = model(data, data_aux.cuda(non_blocking=True))
                else:
                    outputs = model(data)

                if aux_loss:
                    loss = criterion(outputs, label)
                else:
                    target = label.argmax(dim=1).long() if label.ndim == 4 else label.long()
                    loss = criterion(outputs, target)

            # ★ 混合精度反向：用 scaler 包裹
            optimizer.zero_grad(set_to_none=True)  # ★ set_to_none=True 比默认更省显存
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            step += 1
            loss_record.append(loss.item())
            tq.update(batch_size_cur)  # 进度条刷新
            if (i + 1) % 20 == 0 or (i + 1) == len(dataloader_train):
                tq.set_postfix(loss='%.6f' % loss.item())  # 减少终端刷新开销
        tq.close()

        loss_train_mean = np.mean(loss_record)
        print('loss for train :{:.6f}'.format(loss_train_mean))
        writer.add_scalar('{}_loss'.format('train'), loss_train_mean, epoch)

        if (epoch + 1) % args.checkpoint_step == 0 and epoch != 0:
            torch.save(model.state_dict(), os.path.join('checkpoints', args.data_name, args.model_name, datetime,
                                                        'epoch_{:}.pth'.format(epoch + 1)))

        if epoch % args.validation_step == 0:
            if args.num_classes == 2:
                mIoU = val_class2(args, model, dataloader_val, csv_path, epoch,  writer, datetime, loss_train_mean)

            if args.num_classes == 3:
                mIoU = val_class3(args, model, dataloader_val, csv_path, epoch,  writer, datetime, loss_train_mean)

            if args.num_classes == 4:
                mIoU = val_class4(args, model, dataloader_val, csv_path, epoch,  writer, datetime, loss_train_mean)

            if args.num_classes == 5:
                mIoU = val_class5(args, model, dataloader_val, csv_path, epoch,  writer, datetime, loss_train_mean)

            if args.num_classes == 7:
                mIoU = val_class7(args, model, dataloader_val, csv_path, epoch,  writer, datetime, loss_train_mean)

            if mIoU > mIoU_cache:
                torch.save(model.state_dict(), os.path.join('checkpoints', args.data_name, args.model_name, datetime,
                                                            'miou_{:}_epoch_{:}.pth'.format(mIoU, epoch)))
                mIoU_cache = mIoU

    torch.save(model.state_dict(), os.path.join('checkpoints', args.data_name, args.model_name, datetime, 'last.pth'))
    writer.close()
