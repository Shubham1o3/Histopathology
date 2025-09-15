import os
import copy
import pickle
import argparse
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from utils.dataloader import MILDataset
from models.ABMIL import ABMIL
from utils.train import *


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load datasets
    loader_kwargs = {'batch_size': 1, 'num_workers': 2, 'pin_memory': False}
    train_dataset = MILDataset(args.feature_path, args.csv_path, which_split='train', which_labelcol=args.label_col)
    val_dataset = MILDataset(args.feature_path, args.csv_path, which_split='val', which_labelcol=args.label_col)
    test_dataset = MILDataset(args.feature_path, args.csv_path, which_split='test', which_labelcol=args.label_col)

    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    # Model
    model = ABMIL(input_dim=args.feature_dim, hidden_dim=64).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=5)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    all_train_logs, all_val_logs = [], []
    best_model, lowest_val_loss, counter = None, np.inf, 0

    for epoch in range(args.epochs):
        print("--epoch--", epoch)
        train_log = traineval_epoch(epoch, model, train_loader, optimizer=optimizer,
                                    split='train', device=device)
        val_log = traineval_epoch(epoch, model, val_loader, optimizer=None,
                                split='val', device=device)

        all_train_logs.append(train_log)
        all_val_logs.append(val_log)

        val_loss = val_log['val loss']
        
        if val_loss < lowest_val_loss:
            lowest_val_loss, counter, best_model = val_loss, 0, copy.deepcopy(model)
        else:
            counter += 1
            if counter >= args.patience:
                print("Early stopping!")
                break
        
        scheduler.step(val_loss)
        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Epoch {epoch} - LR of group {i}: {param_group['lr']}")
    # Evaluate on test
    test_log = traineval_epoch(epoch, best_model, test_loader, split='test', device=device)
    print("Final Test Results:", test_log)

    # Save model
    torch.save(best_model.state_dict(), args.save_model)


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_path", type=str, required=True, help="Path to .pt features directory")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--label_col", type=str, default="Binary_label")
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--save_model", type=str, default="abmil_model.ckpt")
    args = parser.parse_args()

    main(args)