import torch
import pandas as pd
import torch.nn as nn
import os
import random

class MILDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, feats_dirpath, csv_fpath, which_split='train', which_labelcol='Binary_label'):
        self.feats_dirpath, self.csv, self.which_labelcol = feats_dirpath, pd.read_csv(csv_fpath), which_labelcol
        self.csv_split = self.csv[self.csv['split']==which_split]

    def __getitem__(self, index):
        # Get first sample
        row1 = self.csv_split.iloc[index]
        features1 = torch.load(os.path.join(self.feats_dirpath, row1['slide_id'] + '.pt'))
        label1 = row1[self.which_labelcol]
        slide_name = self.csv_split.iloc[index]['slide_id']
        
        # Get second sample randomly (you can customize sampling strategy)
        index2 = random.randint(0, len(self.csv_split) - 1)
        row2 = self.csv_split.iloc[index2]
        features2 = torch.load(os.path.join(self.feats_dirpath, row2['slide_id'] + '.pt'))
        label2 = row2[self.which_labelcol]
        
        return (features1, label1), (features2, label2),slide_name


    def __len__(self):
        return self.csv_split.shape[0]
    
    def get_label_distribution(self):
        return self.csv_split[self.which_labelcol].value_counts()