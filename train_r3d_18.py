# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import os
import pickle
import logging
import random
import argparse
import traceback
from torchsummary import summary

from dataset import PEDataset
from utils import Timer


# -

def main(args, logger):
    print = logger.info
    print(args)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
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
        
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]))

    print('')
    print("Use cuda " + str(args.gpu))
    print('')
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = criterion.to(device)
    print("Model:")
    print(summary(model, input_size=(3, 10, 192, 192), batch_size=8, device=device, logger=logger))
    print('')

    train_losses_epoch = []
    train_accs_epoch = []
    train_aucs_epoch = []
    val_losses_epoch = []
    val_accs_epoch = []
    val_aucs_epoch = []

    best_auc = 0

    timer = Timer()

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        # training
        print('Training:')
        model.train()

        running_loss_epoch = 0.0
        running_corrects_epoch = 0
        total_samples = 0
        y_trues_epoch = []
        y_scores_epoch = []
        
        timer.start()

        for batch, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)  # N x C x D x H x W
            labels = labels.to(device)

            logit = model(inputs)
            loss = criterion(logit.squeeze(-1), labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred_probs = logit.detach().squeeze(-1).sigmoid().cpu().numpy()
                running_loss_epoch += loss.item() * inputs.size(0)
                running_corrects_epoch += sum((pred_probs >= 0.5) == labels.cpu().numpy())
                total_samples += inputs.size(0)
                y_trues_epoch.extend(labels.cpu().numpy())
                y_scores_epoch.extend(pred_probs)

            if batch % 100 == 99:
                print('{} iters trained | time: {:.2f}'.format(batch+1, timer.stop()))
                timer.start()

        timer.stop()

        # report epoch loss, accuracy, and AUC
        train_loss = running_loss_epoch / total_samples
        train_acc = running_corrects_epoch / total_samples
        train_auc = roc_auc_score(y_trues_epoch, y_scores_epoch)
        print('Train: Loss {:.4f} | Acc {:.4f} | AUC {:.4f}'.format(train_loss, train_acc, train_auc))

        train_losses_epoch.append(train_loss)
        train_accs_epoch.append(train_acc)
        train_aucs_epoch.append(train_auc)

        # evaluation
        print('Validation:')
        model.eval()

        running_loss_epoch = 0.0
        running_corrects_epoch = 0
        total_samples = 0
        y_trues_epoch = []
        y_scores_epoch = []
        
        timer.start()

        for batch, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                logit = model(inputs)
                loss = criterion(logit.squeeze(-1), labels)

                pred_probs = logit.detach().squeeze(-1).sigmoid().cpu().numpy()
                running_loss_epoch += loss.item() * inputs.size(0)
                running_corrects_epoch += sum((pred_probs >= 0.5) == labels.cpu().numpy())
                total_samples += inputs.size(0)
                y_trues_epoch.extend(labels.cpu().numpy())
                y_scores_epoch.extend(pred_probs)

            # report loss, accuracy, and AUC every 100 iterations
            if batch % 100 == 99:
                print('{} iters evaluated | time: {:.2f}'.format(batch+1, timer.stop()))
                timer.start()

        timer.stop()

        # report epoch loss, accuracy, and AUC
        val_loss = running_loss_epoch / total_samples
        val_acc = running_corrects_epoch / total_samples
        val_auc = roc_auc_score(y_trues_epoch, y_scores_epoch)
        print('Val: Loss {:.4f} | Acc {:.4f} | AUC {:.4f}'.format(val_loss, val_acc, val_auc))

        val_losses_epoch.append(val_loss)
        val_accs_epoch.append(val_acc)
        val_aucs_epoch.append(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
            best_epoch = epoch

        print('')

        out_dir = args.model_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(model.state_dict(), out_dir+'/epoch{}.pth'.format(epoch))

    print('Best epoch: {} | Best AUC: {:.4f}'.format(best_epoch, best_auc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="train_r3d_18")
    parser.add_argument("--path", type=str, default="data/train/*/*")
    parser.add_argument("--load_model", type=str, default='')
    parser.add_argument("--gpu", type=int, default=0, help="select gpu to use")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--pos_weight", type=float, default=1.)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--model_dir", type=str, default="weights/r3d_18_0")
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
