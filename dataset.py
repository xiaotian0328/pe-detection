# +
import torch
from torch import nn
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
import os
import math
import random
from tqdm import tqdm
import pickle

from utils import apply_window, apply_transform_3d


# -

class PEDataset(Dataset):
    def __init__(self, paths, mode='train', chunk_size=10, transform=None, transform_3d=None):
        """
        paths: paths of original dcm data
        mode: 'train', 'val', or 'test'
        chunk_size: # of slices in each chunk
        transform: transformation for each slice
        transform_3d: transformation for each chunk
        """
        self.paths = paths
        self.mode = mode
        self.chunk_size = chunk_size
        self.transform = transform
        self.transform_3d = transform_3d

        self.list_chunks = []

        for path in tqdm(self.paths):
            path = 'data/pkl/train/' + path.split('_')[0] + '/' + path.split('_')[1]
            with open(os.path.join(path, 'num_slices.pkl'), 'rb') as f:
                num_slices = pickle.load(f)

            for i in range(num_slices // self.chunk_size):
                chunk_lower = i * self.chunk_size
                chunk_upper = (i + 1) * self.chunk_size
                chunk = [os.path.join(path, str(i)+'.pkl') for i in range(chunk_lower, chunk_upper)]
                self.list_chunks.append(chunk)

            # padding
            if num_slices % self.chunk_size != 0:
                chunk_lower = (num_slices // self.chunk_size) * self.chunk_size
                chunk_upper = num_slices
                chunk = [os.path.join(path, str(i)+'.pkl') for i in range(chunk_lower, chunk_upper)]
                padding = [None] * (self.chunk_size - len(chunk))
                chunk.extend(padding)
                self.list_chunks.append(chunk)

    def __len__(self):
        return len(self.list_chunks)

    def __getitem__(self, idx):
        chunk_paths = self.list_chunks[idx]
        list_images = []
        list_labels = []
        for path in chunk_paths:
            if path is None and self.mode == 'train':
                list_images.append(torch.zeros(3, 224, 224))
            if path is None and (self.mode == 'val' or self.mode == 'test'):
                list_images.append(torch.zeros(3, 192, 192))
            if path is not None:
                with open(path, 'rb') as f:
                    image, label = pickle.load(f)
                # add channels
                image1 = apply_window(image, -600, 1500)
                image2 = apply_window(image, 100, 700)
                image3 = apply_window(image, 40, 400)
                image = np.stack([image1, image2, image3], axis=-1)  # H x W x C
                image = self.transform(image)  # C x H x W
                list_images.append(image)
                list_labels.append(label)
        image = torch.stack(list_images, dim=1)  # C x D x H x W
        if self.transform_3d is not None:
            image = apply_transform_3d(image, self.transform_3d)
        label = float(1 in list_labels)
        return image, label

    def get_num_exams(self):
        return len(self.paths)


class PEFeatureSequenceDataset(Dataset):
    def __init__(self, 
                 paths, 
                 feature_dir,
                 mode='train',  
                 chunk_size=8, 
                 seq_len=32,
                 diff=True):
        self.paths = paths
        self.mode = mode
        self.chunk_size = chunk_size
        self.seq_len = seq_len
        self.diff = diff

        # load features numpy array (N x 512)
        if self.mode == 'train':
            features = np.load(os.path.join(feature_dir, 'train_features.npy'))
        elif self.mode == 'val':
            features = np.load(os.path.join(feature_dir, 'val_features.npy'))
        elif self.mode == 'test':
            features = np.load(os.path.join(feature_dir, 'test_features.npy'))

        data_fields = pd.read_csv('data/train.csv')

        # lists of 9 exam level labels
        self.list_negative_exam_for_pe = []
        self.list_indeterminate = []
        self.list_chronic_pe = []
        self.list_acute_and_chronic_pe = []
        self.list_central_pe = []
        self.list_leftsided_pe = []
        self.list_rightsided_pe = []
        self.list_rv_lv_ratio_gte_1 = []
        self.list_rv_lv_ratio_lt_1 = []

        self.list_series = []  # list of series of chunks for each exam
        self.list_pe_present_series = []  # list of series of chunk level labels

        # iterate each exam
        start = 0
        for path in tqdm(self.paths):
            # extract exam level labels from csv file
            study_instance_uid = path.split('_')[0]
            data_fields_row = data_fields.loc[data_fields['StudyInstanceUID'] == study_instance_uid].iloc[0]

            self.list_negative_exam_for_pe.append(data_fields_row['negative_exam_for_pe'])
            self.list_indeterminate.append(data_fields_row['indeterminate'])
            self.list_chronic_pe.append(data_fields_row['chronic_pe'])
            self.list_acute_and_chronic_pe.append(data_fields_row['acute_and_chronic_pe'])
            self.list_central_pe.append(data_fields_row['central_pe'])
            self.list_leftsided_pe.append(data_fields_row['leftsided_pe'])
            self.list_rightsided_pe.append(data_fields_row['rightsided_pe'])
            self.list_rv_lv_ratio_gte_1.append(data_fields_row['rv_lv_ratio_gte_1'])
            self.list_rv_lv_ratio_lt_1.append(data_fields_row['rv_lv_ratio_lt_1'])

            # load the slice number from pickle file
            path = 'data/pkl/train/' + path.split('_')[0] + '/' + path.split('_')[1]
            with open(os.path.join(path, 'num_slices.pkl'), 'rb') as f:
                num_slices = pickle.load(f)

            # save series of chunks for this exam
            list_pe_present_on_chunk = []
            for i in range(num_slices // self.chunk_size):
                chunk_lower = i * self.chunk_size
                chunk_upper = (i + 1) * self.chunk_size
                list_pe_present = []  # list of whether pe present on each slice
                for i in range(chunk_lower, chunk_upper):
                    with open(os.path.join(path, str(i)+'.pkl'), 'rb') as f:
                        _, label = pickle.load(f)
                        list_pe_present.append(label)
                list_pe_present_on_chunk.append(int(1 in list_pe_present))
            if num_slices % self.chunk_size != 0:
                chunk_lower = (num_slices // self.chunk_size) * self.chunk_size
                chunk_upper = num_slices
                list_pe_present = []  # list of whether pe present on each slice
                for i in range(chunk_lower, chunk_upper):
                    with open(os.path.join(path, str(i)+'.pkl'), 'rb') as f:
                        _, label = pickle.load(f)
                        list_pe_present.append(label)
                list_pe_present_on_chunk.append(int(1 in list_pe_present))

            num_chunks = num_slices // self.chunk_size
            if num_slices % self.chunk_size != 0:
                num_chunks += 1
            end = start + num_chunks
            self.list_series.append(features[start:end])
            self.list_pe_present_series.append(list_pe_present_on_chunk)
            start = end

    def __len__(self):
        return len(self.list_series)

    def __getitem__(self, idx):
        # series of chunk features in the exam
        series = self.list_series[idx]  # numpy array (N x 512)
        # series of labels identify whether pe present in each chunk in the exam
        pe_present_series = self.list_pe_present_series[idx]

        ratio = 3 if self.diff else 1

        if len(series) > self.seq_len:
            # sequential feature embeddings
            x = np.zeros((len(series), series.shape[1]*ratio), dtype=np.float32)
            # labels of whether pe is true for each embedding
            y_pe = np.zeros((len(series), 1), dtype=np.float32)
            # mask for sequence data
            mask = np.ones((self.seq_len, ), dtype=np.float32)
            for i in range(len(series)):
                x[i, :series.shape[1]] = series[i]
                y_pe[i] = pe_present_series[i]
            # resize the sequence length to seq_len
            x = cv2.resize(x, (series.shape[1]*ratio, self.seq_len), interpolation=cv2.INTER_LINEAR)
            y_pe = np.squeeze(cv2.resize(y_pe, (1, self.seq_len), interpolation=cv2.INTER_LINEAR))
        else:
            # sequential feature embeddings
            x = np.zeros((self.seq_len, series.shape[1]*ratio), dtype=np.float32)
            # labels of whether pe is true for each embedding
            y_pe = np.zeros((self.seq_len, ), dtype=np.float32)
            # mask for sequence data
            mask = np.zeros((self.seq_len, ), dtype=np.float32)
            for i in range(len(series)):
                x[i, :series.shape[1]] = series[i]
                y_pe[i] = pe_present_series[i]
                mask[i] = 1.

        if self.diff:
            # concatenate the difference between two neighbor embeddings
            x[1:, series.shape[1]:series.shape[1]*2] = x[1:, :series.shape[1]] - x[:-1, :series.shape[1]]
            x[:-1, series.shape[1]*2:] = x[:-1, :series.shape[1]] - x[1:, :series.shape[1]]

        y_negative_exam_for_pe = self.list_negative_exam_for_pe[idx]
        y_indeterminate = self.list_indeterminate[idx]
        y_chronic_pe = self.list_chronic_pe[idx]
        y_acute_and_chronic_pe = self.list_acute_and_chronic_pe[idx]
        y_central_pe = self.list_central_pe[idx]
        y_leftsided_pe = self.list_leftsided_pe[idx]
        y_rightsided_pe = self.list_rightsided_pe[idx]
        y_rv_lv_ratio_gte_1 = self.list_rv_lv_ratio_gte_1[idx]
        y_rv_lv_ratio_lt_1 = self.list_rv_lv_ratio_lt_1[idx]

        # proportion of positive chunks in the exam
        q_i = np.float32(sum(pe_present_series) / len(series))

        # x: (seq_len, 512 x ratio)
        # mask: (seq_len, )
        # y_pe: (seq_len, )
        return x, mask, \
               y_pe, \
               y_negative_exam_for_pe, \
               y_indeterminate, \
               y_chronic_pe, \
               y_acute_and_chronic_pe, \
               y_central_pe, \
               y_leftsided_pe, \
               y_rightsided_pe, \
               y_rv_lv_ratio_gte_1, \
               y_rv_lv_ratio_lt_1, \
               q_i, len(series)
