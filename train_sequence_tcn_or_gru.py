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
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from apex import amp
import pandas as pd
import numpy as np
import os
import random
import argparse
import pickle
import logging
import traceback
from torchsummary import summary

from dataset import PEFeatureSequenceDataset
from model import PEFeatureSequentialNet
from utils import Timer, AverageMeter


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
    print("train, val, test split: train {}, val {}, test {}"\
          .format(len(train_paths), len(val_paths), len(test_paths)))

    train_set = PEFeatureSequenceDataset(paths=train_paths, 
                                         feature_dir=args.feature_dir, 
                                         mode='train', 
                                         chunk_size=args.chunk_size,
                                         seq_len=args.seq_len, 
                                         diff=args.diff)

    val_set = PEFeatureSequenceDataset(paths=val_paths, 
                                       mode='val', 
                                       feature_dir=args.feature_dir, 
                                       chunk_size=args.chunk_size,
                                       seq_len=args.seq_len, 
                                       diff=args.diff)

    print("train set length: {} | val set length: {}".format(len(train_set), len(val_set)))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
    print("train loader iters: {} | val loader iters: {}".format(len(train_loader), len(val_loader)))

    if args.load_model == '':
        model = PEFeatureSequentialNet(seq_model=args.seq_model, 
                                       input_size=args.feature_size,
                                       hidden_size=args.hidden_size,
                                       levels=args.levels, 
                                       kernel_size=args.kernel_size, 
                                       seq_len=args.seq_len,
                                       dropout=args.dropout, 
                                       bidirectional=args.bidirectional, 
                                       maxpool=args.maxpool, 
                                       diff=args.diff, 
                                       batchnorm=args.batchnorm)
    else:
        model = PEFeatureSequentialNet(seq_model=args.seq_model, 
                                       input_size=args.feature_size,
                                       hidden_size=args.hidden_size,
                                       levels=args.levels, 
                                       kernel_size=args.kernel_size, 
                                       seq_len=args.seq_len,
                                       dropout=args.dropout, 
                                       bidirectional=args.bidirectional, 
                                       maxpool=args.maxpool, 
                                       diff=args.diff, 
                                       batchnorm=args.batchnorm)
        model.load_state_dict(torch.load("weights/"+args.load_model))
        
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
    criterion = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor([args.pos_weight]))
    criterion_exam_level = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.pos_weight]))

    print('')
    print("Use cuda " + str(args.gpu))
    print('')
    device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    model.to(device)
    criterion.to(device)
    criterion_exam_level.to(device)

    # acceleration
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    label_list = [
        'pe_present_on_image',
        'negative_exam_for_pe',
        'indeterminate',
        'chronic_pe',
        'acute_and_chronic_pe',
        'central_pe',
        'leftsided_pe',
        'rightsided_pe',
        'rv_lv_ratio_gte_1',
        'rv_lv_ratio_lt_1'
    ]

    loss_weight_dict = {
        'pe_present_on_image': 0.07361963,
        'negative_exam_for_pe': 0.0736196319,
        'indeterminate': 0.09202453988,
        'chronic_pe': 0.1042944785,
        'acute_and_chronic_pe': 0.1042944785,
        'central_pe': 0.1877300613,
        'leftsided_pe': 0.06257668712,
        'rightsided_pe': 0.06257668712,
        'rv_lv_ratio_gte_1': 0.2346625767,
        'rv_lv_ratio_lt_1': 0.0782208589
    }
    
    best_loss = float('inf')

    timer = Timer()

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        # training
        print('Training:')
        model.train()

        losses = {label: AverageMeter() for label in label_list}
        losses['total'] = AverageMeter()
        accs = {label: AverageMeter() for label in label_list}
        aucs = {label: 0. for label in label_list}
        y_trues = {label: [] for label in label_list}
        y_scores = {label: [] for label in label_list}

        timer.start()

        for batch, data in enumerate(train_loader):
            x, mask = data[0], data[1]
            labels = {label_list[i]:data[i+2] for i in range(len(label_list))}
            q_i, series_len = data[-2], data[-1]  # q_i, series_len: (batch_size, )

            adjustment = [series_len[i].item() / args.seq_len if series_len[i] > args.seq_len else 1. for i in range(x.size(0))]
            adjustment = torch.tensor(adjustment, dtype=torch.float32)  # (batch_size, )
            loss_weight_pe = torch.tensor(loss_weight_dict['pe_present_on_image'], dtype=torch.float32)
            loss_pe_weights = loss_weight_pe * q_i * adjustment  # (batch_size, )
            loss_pe_weights = loss_pe_weights.unsqueeze(1).to(device)  # (batch_size, 1)

            x = x.to(device)  # (batch_size, seq_len, feature_size)
            mask = mask.to(device)  # (batch_size, seq_len)
            for key in label_list:
                labels[key] = labels[key].float().to(device)

            logits = model(x, mask)
#             logits = {label_list[i]:logit_values[i] for i in range(len(label_list))}

            loss_dict = {label: 0. for label in label_list}

            # chunk level loss
            # logits['pe_present_on_image']: (batch_size, seq_len, 1)
            # labels['pe_present_on_image']: (batch_size, seq_len)
            loss_dict['pe_present_on_image'] = criterion(logits['pe_present_on_image'].squeeze(-1),
                                                         labels['pe_present_on_image'])  # (batch_size, seq_len)
            loss_dict['pe_present_on_image'] = loss_dict['pe_present_on_image'] * mask * loss_pe_weights  # (batch_size, seq_len)
            loss_dict['pe_present_on_image'] = loss_dict['pe_present_on_image'].sum() / mask.sum()

            # exam level loss
            for key in label_list[1:]:
                loss_dict[key] = criterion_exam_level(logits[key].view(-1), labels[key]) * loss_weight_dict[key]

            loss = 0
            for key in label_list:
                loss += loss_dict[key]

            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            total_loss = 0
            total_weight = 0
            losses['pe_present_on_image'].update(loss_dict['pe_present_on_image'].item(), mask.sum().item())
            total_loss += loss_dict['pe_present_on_image'].item() * mask.sum().item()
            total_weight += (mask * loss_pe_weights).sum().item()
            for key in label_list[1:]:
                losses[key].update(loss_dict[key].item(), x.size(0))
                total_loss += loss_dict[key].item() * x.size(0)
                total_weight += loss_weight_dict[key] * x.size(0)
            losses['total'].update(total_loss / total_weight, total_weight)

            # exam level evaluations
            for key in label_list[1:]:
                pred_prob = np.squeeze(logits[key].detach().sigmoid().cpu().numpy())
                label = labels[key].detach().cpu().numpy()
                accs[key].update(sum((pred_prob > 0.5) == label) / x.size(0), x.size(0))
                y_scores[key].extend(pred_prob.tolist())
                y_trues[key].extend(label.tolist())

        timer.stop()

        # calculate AUCs
        for key in label_list[1:]:
            try:
                aucs[key] = roc_auc_score(y_trues[key], y_scores[key])
            except:
                pass

        # report losses, accuracies, AUCs of each epoch
        for key in label_list[1:]:
            print('{:>20}: loss {:.4f} | Acc {:.4f} | AUC {:.4f}'.format(key, 
                                                                         losses[key].avg,
                                                                         accs[key].avg, 
                                                                         aucs[key]))
        print('Total loss: {:4f}'.format(losses['total'].avg))
        

        print('-' * 10)

        # evaluation
        print('Validation:')
        model.eval()

        losses = {label: AverageMeter() for label in label_list}
        losses['total'] = AverageMeter()
        accs = {label: AverageMeter() for label in label_list}
        aucs = {label: 0. for label in label_list}
        y_trues = {label: [] for label in label_list}
        y_scores = {label: [] for label in label_list}

        timer.start()

        for batch, data in enumerate(val_loader):
            with torch.no_grad():
                x, mask = data[0], data[1]
                labels = {label_list[i]:data[i+2] for i in range(len(label_list))}
                q_i, series_len = data[-2], data[-1]  # q_i, series_len: (batch_size, )

                adjustment = [series_len[i].item() / args.seq_len if series_len[i] > args.seq_len else 1. for i in range(x.size(0))]
                adjustment = torch.tensor(adjustment, dtype=torch.float32)  # (batch_size, )
                loss_weight_pe = torch.tensor(loss_weight_dict['pe_present_on_image'], dtype=torch.float32)
                loss_pe_weights = loss_weight_pe * q_i * adjustment  # (batch_size, )
                loss_pe_weights = loss_pe_weights.unsqueeze(1).to(device)  # (batch, 1)

                x = x.to(device)  # (batch, seq_len, feature_size)
                mask = mask.to(device)  # (batch, seq_len)
                for key in label_list:
                    labels[key] = labels[key].float().to(device)

                logits = model(x, mask)

                loss_dict = {label: 0. for label in label_list}

                # chunk level loss
                # logits['pe_present_on_image']: (batch_size, seq_len, 1)
                # labels['pe_present_on_image']: (batch_size, seq_len)
                loss_dict['pe_present_on_image'] = criterion(logits['pe_present_on_image'].squeeze(-1), 
                                                             labels['pe_present_on_image'])  # (batch_size, seq_len)
                loss_dict['pe_present_on_image'] = loss_dict['pe_present_on_image'] * mask * loss_pe_weights
                loss_dict['pe_present_on_image'] = loss_dict['pe_present_on_image'].sum() / mask.sum()

                # exam level loss
                for key in label_list[1:]:
                    loss_dict[key] = criterion_exam_level(logits[key].view(-1), labels[key]) * loss_weight_dict[key]

                total_loss = 0
                total_weight = 0
                losses['pe_present_on_image'].update(loss_dict['pe_present_on_image'].item(), mask.sum().item())
                total_loss += loss_dict['pe_present_on_image'].item() * mask.sum().item()
                total_weight += (mask * loss_pe_weights).sum().item()
                for key in label_list[1:]:
                    losses[key].update(loss_dict[key].item(), x.size(0))
                    total_loss += loss_dict[key].item() * x.size(0)
                    total_weight += loss_weight_dict[key] * x.size(0)
                losses['total'].update(total_loss / total_weight, total_weight)

                # exam level evaluations
                for key in label_list[1:]:
                    pred_prob = np.squeeze(logits[key].detach().sigmoid().cpu().numpy())
                    label = labels[key].detach().cpu().numpy()
                    accs[key].update(sum((pred_prob > 0.5) == label) / x.size(0), x.size(0))
                    y_scores[key].extend(pred_prob.tolist())
                    y_trues[key].extend(label.tolist())

        timer.stop()

        # calculate AUCs
        for key in label_list[1:]:
            try:
                aucs[key] = roc_auc_score(y_trues[key], y_scores[key])
            except:
                pass

        # report losses, accuracies, AUCs of each epoch
        for key in label_list[1:]:
            print('{:>20}: loss {:.4f} | Acc {:.4f} | AUC {:.4f}'.format(key, 
                                                                         losses[key].avg,
                                                                         accs[key].avg, 
                                                                         aucs[key]))
        print('Total loss: {:4f}'.format(losses['total'].avg))
        
        if losses['total'].avg < best_loss:
            best_loss = losses['total'].avg
            best_epoch = epoch

        print('-' * 10)
        print('')

        out_dir = "weights/" + args.model_dir
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        torch.save(model.state_dict(), out_dir+'/epoch{}.pth'.format(epoch))
        
        scheduler.step()

    print('Best epoch: {} | Best loss: {:.4f}'.format(best_epoch, best_loss))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="train_sequence")
    parser.add_argument("--path", type=str, default="data/train/*/*")
    parser.add_argument("--load_model", type=str, default='')
    parser.add_argument("--gpu", type=int, default=0, help="select gpu to use")
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--pos_weight", type=float, default=5.)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=10)
    parser.add_argument("--feature_dir", type=str, default="features/r3d_18")
    parser.add_argument("--seq_model", type=str, default="TCN")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--feature_size", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--levels", type=int, default=2)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--model_dir", type=str, default="sequence")

    parser.add_argument("--diff", dest="diff", action="store_true")
    parser.add_argument("--no-diff", dest="diff", action="store_false")
    parser.set_defaults(diff=False)

    parser.add_argument("--maxpool", dest="maxpool", action="store_true")
    parser.add_argument("--no-maxpool", dest="maxpool", action="store_false")
    parser.set_defaults(maxpool=False)

    parser.add_argument("--batchnorm", dest="batchnorm", action="store_true")
    parser.add_argument("--no-batchnorm", dest="batchnorm", action="store_false")
    parser.set_defaults(batchnorm=False)

    parser.add_argument("--bidirectional", dest="bidirectional", action="store_true")
    parser.add_argument("--no-bidirectional", dest="bidirectional", action="store_false")
    parser.set_defaults(bidirectional=False)

    args = parser.parse_args()

    logfile = 'logs/' + args.log_file
    if os.path.exists(logfile): 
        os.system('rm {}'.format(logfile))
    logging.basicConfig(level=logging.INFO, 
                        format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S')
    logger = logging.getLogger('root')
    logger.addHandler(logging.FileHandler(logfile, 'a'))

    try:
        main(args, logger)
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
