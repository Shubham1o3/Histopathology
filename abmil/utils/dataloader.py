import torch
import pandas as pd
import torch.nn as nn
import os

class MILDataset(torch.utils.data.Dataset):
    def __init__(self, feats_dirpath, csv_fpath, which_split='train', which_labelcol='Binary_label'):
        self.feats_dirpath, self.csv, self.which_labelcol = feats_dirpath, pd.read_csv(csv_fpath), which_labelcol
        self.csv_split = self.csv[self.csv['split'] == which_split]

    def __getitem__(self, index):
        features = torch.load(os.path.join(self.feats_dirpath, self.csv_split.iloc[index]['slide_id'] + '.pt'))
        label = self.csv_split.iloc[index][self.which_labelcol]
        slide_name = self.csv_split.iloc[index]['slide_id']
        return features, label, slide_name

    def __len__(self):
        return self.csv_split.shape[0]