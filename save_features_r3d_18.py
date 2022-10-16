# +
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
import logging
import math
import random
import pickle
import argparse
import traceback
from torchsummary import summary

from dataset import PEDataset
from utils import Timer


# -

def main(args, logger):
    print = logger.info
    print(args)

    with open('data/data_splits.pkl', 'rb') as f:
        data_splits = pickle.load(f)
    train_paths = data_splits['train_paths']
    val_paths = data_splits['val_paths']
    test_paths = data_splits['test_paths']
    print("train, val, test split: train {}, val {}, test{}"\
          .format(len(train_paths), len(val_paths), len(test_paths)))

    transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
    transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((192, 192)),
        ])

    # https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p2ch12/training.py
    # line 105 - 115
    transform_3d_train = {
        'crop': 192,
        'rotate': 15,
    }

    train_set = PEDataset(paths=train_paths, mode='train', chunk_size=args.chunk_size,
                          transform=transform_train, transform_3d=transform_3d_train)
    val_set = PEDataset(paths=val_paths, mode='val', chunk_size=args.chunk_size,
                        transform=transform_val)

    print("train exams: {} | val exams: {}".format(train_set.get_num_exams(), val_set.get_num_exams()))
    print("train chunks: {} | val chunks: {}".format(len(train_set), len(val_set)))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    print("train loader iters: {} | val loader iters: {}".format(len(train_loader), len(val_loader)))

    if args.load_model == '':
        model = models.video.r3d_18(pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
    else:
        model = models.video.r3d_18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 1)
        model.load_state_dict(torch.load("weights/"+args.load_model))

    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))

    print('')
    print("Use cuda " + str(args.gpu))
    print('')
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    feature_extractor.to(device)
    print("Feature Extractor:")
    print(summary(feature_extractor, input_size=(3, 10, 192, 192), batch_size=args.batch_size, device=device, logger=logger))
    print('')

    train_size = len(train_set)
    val_size = len(val_set)
    
    timer = Timer()

    feature_extractor.eval()

    train_features = np.zeros((train_size, 512))
    train_labels = np.zeros((train_size, ))

    timer.start()

    for batch, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            feature = feature_extractor(inputs)
            feature = feature.view(inputs.size(0), -1)
            start= batch * args.batch_size
            end = start + inputs.size(0)
            train_features[start:end] = feature.cpu().numpy()
            train_labels[start:end] = labels.numpy()

        # report loss, accuracy, and AUC every 100 iterations
        if batch % 100 == 99:
            print('{} iters | time: {:.2f}'.format(batch+1, timer.stop()))
            timer.start()

    timer.stop()

    print('')

    out_dir = args.feature_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(out_dir+'/train_features', train_features)
    np.save(out_dir+'/train_labels', train_labels)

    val_features = np.zeros((val_size, 512))
    val_labels = np.zeros((val_size, ))

    timer.start()

    for batch, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        with torch.no_grad():
            feature = feature_extractor(inputs)
            feature = feature.view(inputs.size(0), -1)
            start= batch * args.batch_size
            end = start + inputs.size(0)
            val_features[start:end] = feature.cpu().numpy()
            val_labels[start:end] = labels.numpy()

        # report loss, accuracy, and AUC every 100 iterations
        if batch % 100 == 99:
            print('{} iters | time: {:.2f}'.format(batch+1, timer.stop()))
            timer.start()

    timer.stop()

    print('')

    out_dir = args.feature_dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    np.save(out_dir+'/val_features', val_features)
    np.save(out_dir+'/val_labels', val_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="save_features_r3d_18")
    parser.add_argument("--path", type=str, default="data/train/*/*")
    parser.add_argument("--load_model", type=str, default="r3d_18")
    parser.add_argument("--gpu", type=int, default=0, help="select gpu to use")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--feature_dir", type=str, default="features/r3d_18")
    args = parser.parse_args()

    logfile = 'logs/' + args.log_file
    if os.path.exists(logfile): 
        os.system('rm {}'.format(logfile))
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S')
    logger = logging.getLogger('root')
    logger.addHandler(logging.FileHandler(logfile, 'a'))

    try:
        main(args, logger)
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc)
