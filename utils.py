import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


#########_____return label -> {label_name: [r_value, g_value, b_value, ...}______######
def get_label_info(csv_path):
    ann = pd.read_csv(csv_path)
    label = {}
    for iter, row in ann.iterrows():
        label_name = row['name']  # 把row中的数据赋值给label字典
        r = row['r']
        g = row['g']
        b = row['b']
        label[label_name] = [int(r), int(g), int(b)]
    return label


##########_______________one_hot_____________#########################
def one_hot_it(label, label_info):
    semantic_map = []
    for info in label_info:  # 遍历label_info中的键
        color = label_info[info]  # 把label_info的键对应的值赋值给color
        equality = np.equal(label, color)  # 判断是否相等
        class_map = np.all(equality, axis=-1)  # 判断某个轴上元素是否全为True
        semantic_map.append(class_map)  #
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=600, power=2.0):  # power= 0.9或者2.0
    """Polynomial decay of learning rate
		:param init_lr is base learning rate
		:param iter is a current iteration
		:param lr_decay_iter how frequently decay occurs, default is 1
		:param max_iter is number of maximum iterations
		:param power is a polymomial power
	"""
    if iter % lr_decay_iter or iter > max_iter:
        return optimizer
    lr = init_lr * (1 - iter / max_iter) ** power  # 随着epoch 的变化，学习率不断改变
    optimizer.param_groups[0]['lr'] = lr  # 把新的lr值给优化器
    return lr


def reverse_one_hot(image):
    """
	Transform a 2D array in one-hot format (depth is num_classes),
	to a 2D array with only 1 channel, where each pixel value is
	the classified class key.

	# Arguments
		image: The one-hot format image

	# Returns
		A 2D array with the same width and height as the input, but
		with a depth size of 1, where each pixel value is the classified
		class key.
	"""
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index
    image = image.permute(1, 2, 0)  # 转换维度，此处将CHW 转换为了HWC
    x = torch.argmax(image, dim=-1)  # 返回指定维度最大值的序号
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    label_values = [label_values[key] for key in label_values]  # 遍历字典，将所有值放在一个列表中
    colour_codes = np.array(label_values)  # 例如 ：  [[128, 0, 0], [0, 128, 0], [0, 0, 0]]
    x = colour_codes[image.astype(int)]  # 用数组索引数组。 其中image是经过处理后的，代表每个像素点颜色的序号。
    return x


def compute_global_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count = count + 1.0
    return float(count) / float(total)


def fast_hist(a, b, n):
    '''
	a and b are predict and mask respectively
	n is the number of classes
	'''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    epsilon = 1e-5
    return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


'''  
     0:object 1:background 2:void
'''


def coutpixel_class7(y_true, y_predict):
    N00 = np.sum((y_true == 0) & (y_predict == 0))
    N01 = np.sum((y_true == 0) & (y_predict == 1))
    N02 = np.sum((y_true == 0) & (y_predict == 2))
    N03 = np.sum((y_true == 0) & (y_predict == 3))
    N04 = np.sum((y_true == 0) & (y_predict == 4))
    N05 = np.sum((y_true == 0) & (y_predict == 5))
    N06 = np.sum((y_true == 0) & (y_predict == 6))

    N10 = np.sum((y_true == 1) & (y_predict == 0))
    N11 = np.sum((y_true == 1) & (y_predict == 1))
    N12 = np.sum((y_true == 1) & (y_predict == 2))
    N13 = np.sum((y_true == 1) & (y_predict == 3))
    N14 = np.sum((y_true == 1) & (y_predict == 4))
    N15 = np.sum((y_true == 1) & (y_predict == 5))
    N16 = np.sum((y_true == 1) & (y_predict == 6))

    N20 = np.sum((y_true == 2) & (y_predict == 0))
    N21 = np.sum((y_true == 2) & (y_predict == 1))
    N22 = np.sum((y_true == 2) & (y_predict == 2))
    N23 = np.sum((y_true == 2) & (y_predict == 3))
    N24 = np.sum((y_true == 2) & (y_predict == 4))
    N25 = np.sum((y_true == 2) & (y_predict == 5))
    N26 = np.sum((y_true == 2) & (y_predict == 6))

    N30 = np.sum((y_true == 3) & (y_predict == 0))
    N31 = np.sum((y_true == 3) & (y_predict == 1))
    N32 = np.sum((y_true == 3) & (y_predict == 2))
    N33 = np.sum((y_true == 3) & (y_predict == 3))
    N34 = np.sum((y_true == 3) & (y_predict == 4))
    N35 = np.sum((y_true == 3) & (y_predict == 5))
    N36 = np.sum((y_true == 3) & (y_predict == 6))

    N40 = np.sum((y_true == 4) & (y_predict == 0))
    N41 = np.sum((y_true == 4) & (y_predict == 1))
    N42 = np.sum((y_true == 4) & (y_predict == 2))
    N43 = np.sum((y_true == 4) & (y_predict == 3))
    N44 = np.sum((y_true == 4) & (y_predict == 4))
    N45 = np.sum((y_true == 4) & (y_predict == 5))
    N46 = np.sum((y_true == 4) & (y_predict == 6))

    N50 = np.sum((y_true == 5) & (y_predict == 0))
    N51 = np.sum((y_true == 5) & (y_predict == 1))
    N52 = np.sum((y_true == 5) & (y_predict == 2))
    N53 = np.sum((y_true == 5) & (y_predict == 3))
    N54 = np.sum((y_true == 5) & (y_predict == 4))
    N55 = np.sum((y_true == 5) & (y_predict == 5))
    N56 = np.sum((y_true == 5) & (y_predict == 6))

    N60 = np.sum((y_true == 6) & (y_predict == 0))
    N61 = np.sum((y_true == 6) & (y_predict == 1))
    N62 = np.sum((y_true == 6) & (y_predict == 2))
    N63 = np.sum((y_true == 6) & (y_predict == 3))
    N64 = np.sum((y_true == 6) & (y_predict == 4))
    N65 = np.sum((y_true == 6) & (y_predict == 5))
    N66 = np.sum((y_true == 6) & (y_predict == 6))

    return N00, N01, N02, N03, N04, N05, N06, N10, N11, N12, N13, N14, N15, N16, N20, N21, N22, N23, N24, N25, N26, N30, N31, N32, N33, N34, N35, N36, N40, N41, N42, N43, N44, N45, N46, N50, N51, N52, N53, N54, N55, N56, N60, N61, N62, N63, N64, N65, N66


def coutpixel_class5(y_true, y_predict):
    N00 = np.sum((y_true == 0) & (y_predict == 0))
    N01 = np.sum((y_true == 0) & (y_predict == 1))
    N02 = np.sum((y_true == 0) & (y_predict == 2))
    N03 = np.sum((y_true == 0) & (y_predict == 3))
    N04 = np.sum((y_true == 0) & (y_predict == 4))

    N10 = np.sum((y_true == 1) & (y_predict == 0))
    N11 = np.sum((y_true == 1) & (y_predict == 1))
    N12 = np.sum((y_true == 1) & (y_predict == 2))
    N13 = np.sum((y_true == 1) & (y_predict == 3))
    N14 = np.sum((y_true == 1) & (y_predict == 4))

    N20 = np.sum((y_true == 2) & (y_predict == 0))
    N21 = np.sum((y_true == 2) & (y_predict == 1))
    N22 = np.sum((y_true == 2) & (y_predict == 2))
    N23 = np.sum((y_true == 2) & (y_predict == 3))
    N24 = np.sum((y_true == 2) & (y_predict == 4))

    N30 = np.sum((y_true == 3) & (y_predict == 0))
    N31 = np.sum((y_true == 3) & (y_predict == 1))
    N32 = np.sum((y_true == 3) & (y_predict == 2))
    N33 = np.sum((y_true == 3) & (y_predict == 3))
    N34 = np.sum((y_true == 3) & (y_predict == 4))

    N40 = np.sum((y_true == 4) & (y_predict == 0))
    N41 = np.sum((y_true == 4) & (y_predict == 1))
    N42 = np.sum((y_true == 4) & (y_predict == 2))
    N43 = np.sum((y_true == 4) & (y_predict == 3))
    N44 = np.sum((y_true == 4) & (y_predict == 4))

    return N00, N01, N02, N03, N04, N10, N11, N12, N13, N14, N20, N21, N22, N23, N24, N30, N31, N32, N33, N34, N40, N41, N42, N43, N44


def coutpixel_class4(y_true, y_predict):
    N00 = np.sum((y_true == 0) & (y_predict == 0))
    N01 = np.sum((y_true == 0) & (y_predict == 1))
    N02 = np.sum((y_true == 0) & (y_predict == 2))
    N03 = np.sum((y_true == 0) & (y_predict == 3))

    N10 = np.sum((y_true == 1) & (y_predict == 0))
    N11 = np.sum((y_true == 1) & (y_predict == 1))
    N12 = np.sum((y_true == 1) & (y_predict == 2))
    N13 = np.sum((y_true == 1) & (y_predict == 3))

    N20 = np.sum((y_true == 2) & (y_predict == 0))
    N21 = np.sum((y_true == 2) & (y_predict == 1))
    N22 = np.sum((y_true == 2) & (y_predict == 2))
    N23 = np.sum((y_true == 2) & (y_predict == 3))

    N30 = np.sum((y_true == 3) & (y_predict == 0))
    N31 = np.sum((y_true == 3) & (y_predict == 1))
    N32 = np.sum((y_true == 3) & (y_predict == 2))
    N33 = np.sum((y_true == 3) & (y_predict == 3))

    return N00, N01, N02, N03, N10, N11, N12, N13, N20, N21, N22, N23, N30, N31, N32, N33


def coutpixel_class3(y_true, y_predict):
    N00 = np.sum((y_true == 0) & (y_predict == 0))
    N01 = np.sum((y_true == 0) & (y_predict == 1))
    N02 = np.sum((y_true == 0) & (y_predict == 2))

    N10 = np.sum((y_true == 1) & (y_predict == 0))
    N11 = np.sum((y_true == 1) & (y_predict == 1))
    N12 = np.sum((y_true == 1) & (y_predict == 2))

    N20 = np.sum((y_true == 2) & (y_predict == 0))
    N21 = np.sum((y_true == 2) & (y_predict == 1))
    N22 = np.sum((y_true == 2) & (y_predict == 2))

    return N00, N01, N02, N10, N11, N12, N20, N21, N22


def coutpixel_class2(y_true, y_predict):
    N00 = np.sum((y_true == 0) & (y_predict == 0))
    N01 = np.sum((y_true == 0) & (y_predict == 1))

    N10 = np.sum((y_true == 1) & (y_predict == 0))
    N11 = np.sum((y_true == 1) & (y_predict == 1))

    return N00, N01, N10, N11


import cv2
import numpy as np


class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targeted intermediate layers """

    def __init__(self, model, target_layers, reshape_transform):
        self.model = model
        self.gradients = []
        self.activations = []
        self.reshape_transform = reshape_transform
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(
                    self.save_activation))
            # Backward compatibility with older pytorch versions:
            if hasattr(target_layer, 'register_full_backward_hook'):
                self.handles.append(
                    target_layer.register_full_backward_hook(
                        self.save_gradient))
            else:
                self.handles.append(
                    target_layer.register_backward_hook(
                        self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        if self.reshape_transform is not None:
            activation = self.reshape_transform(activation)
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        grad = grad_output[0]
        if self.reshape_transform is not None:
            grad = self.reshape_transform(grad)
        self.gradients = [grad.cpu().detach()] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()
