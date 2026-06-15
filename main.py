import argparse
import os
import time
import warnings

import torch
from torch.utils.data import DataLoader

from dataset import HyDataset
from models.specgatenet import SpecGateNet
from train import train

warnings.filterwarnings('ignore')


nets = {
    'SpecGateNet': SpecGateNet(size=224, n_classes=7, cnn_pretrained=True, aux_loss=False),
}


def main(params, model):
    parser = argparse.ArgumentParser()
    default_workers = 2 if os.name == 'nt' else 8
    # dataset arguments
    parser.add_argument('--crop_height', type=int, default=224, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=224, help='Width of cropped/resized input image to network')
    parser.add_argument('--data_root', type=str, default='datasets', help='root path of datasets')
    parser.add_argument('--data_name', type=str, default='SPARCS', help='name of dataset under data_root')
    parser.add_argument('--num_classes', type=int, default=3, help='num of object classes (with void)')
    parser.add_argument('--demo_root', type=str, default='demo', help='root path of demo images')
    parser.add_argument('--demo_name', type=str, default='demo', help='name of demo set')
    parser.add_argument('--aux_bands', type=list, default=[], help='list of aux bands')

    # model arguments
    parser.add_argument('--model_name', type=str, default='SpecGateNet', help='path to save model')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path of pretrained model')
    # train arguments
    parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Number of images in each batch')
    parser.add_argument('--val_batch_size', type=int, default=8, help='Validation batch size for class-3 tasks')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate used for train')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--enable_demo', type=bool, default=False, help='whether to demo on images')
    # run arguments
    parser.add_argument('--num_workers', type=int, default=default_workers, help='num of workers')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to use gpu for training')
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_wandb', type=bool, default=False, help='use wandb to record exp')

    args = parser.parse_args(params)

    # dataset layout: <data_root>/<data_name>/{train,val}/{img,label} + class_dict.csv
    train_root = os.path.join(args.data_root, args.data_name, 'train')
    val_root = os.path.join(args.data_root, args.data_name, 'val')
    csv_path = os.path.join(args.data_root, args.data_name, 'class_dict.csv')

    dataset_train = HyDataset(data_root=train_root, csv_path=csv_path, scale=(args.crop_height, args.crop_width),
                              mode='train', aux_bands=args.aux_bands)
    train_loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    if args.num_workers > 0:
        train_loader_kwargs['persistent_workers'] = True
        train_loader_kwargs['prefetch_factor'] = 4 if (os.cpu_count() or 0) > 4 else 2
    dataloader_train = DataLoader(dataset_train, **train_loader_kwargs)
    dataset_val = HyDataset(data_root=val_root, csv_path=csv_path, scale=(args.crop_height, args.crop_width),
                            mode='val', aux_bands=args.aux_bands)
    val_loader_kwargs = {
        'batch_size': args.val_batch_size if args.num_classes == 3 else 1,
        'shuffle': False,
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    if args.num_workers > 0:
        val_loader_kwargs['persistent_workers'] = True
        val_loader_kwargs['prefetch_factor'] = 4 if (os.cpu_count() or 0) > 4 else 2
    dataloader_val = DataLoader(dataset_val, **val_loader_kwargs)

    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # load pretrained model if it exists
    if args.pretrained_model_path is not None:
        print('load model from %s...' % args.pretrained_model_path)
        model.load_state_dict(torch.load(args.pretrained_model_path), strict=False)
        print('Done!')

    # start to train
    datetime = time.strftime("%Y_%m_%d_%H_%M_", time.localtime())

    if not os.path.exists(os.path.join('checkpoints', args.data_name, args.model_name, datetime)):
        os.makedirs(os.path.join('checkpoints', args.data_name, args.model_name, datetime))

    if not os.path.exists(os.path.join('result')):
        os.makedirs(os.path.join('result'))

    if args.enable_demo:
        if not os.path.exists(f'{args.demo_root}/{args.model_name}/{datetime}'):
            os.makedirs(f'{args.demo_root}/{args.model_name}/{datetime}')

    train(args, model, dataloader_train, dataloader_val, csv_path, datetime, aux_loss=False)


if __name__ == '__main__':
    net = 'SpecGateNet'                     # model to train
    params = [
        '--epoch_start_i', '0',             # resume: start counting epochs from here
        '--num_epochs', '300',
        '--crop_height', '224',
        '--crop_width', '224',
        '--learning_rate', '0.00005',

        '--data_root', 'datasets',          # <data_root>/<data_name>/...
        '--data_name', 'SPARCS',            # e.g. SPARCS (7 cls) / CloudSEN12 (3 cls) / 38-Cloud (2 cls)

        '--aux_bands', [],                  # optional auxiliary bands (unused by default)

        '--num_workers', '2',
        '--num_classes', '7',               # must match the chosen dataset's class_dict.csv
        '--batch_size', '16',
        '--val_batch_size', '1',
        '--checkpoint_step', '30',
        '--model_name', net,
        '--use_wandb', False,
    ]
    net = nets[net]
    main(params, net)
