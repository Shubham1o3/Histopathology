import os
import copy
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# print(torch.__version__)
import papermill
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau

feature_dimension =1024
which_labelcol = 'Binary_label'
feats_dirpath = '/home/tifr1/ms1/CLAM/FEATURES_DIRECTORY4000/pt_files'
# csv_fpath = '/drive2/shubham_tifr/dataset_split_balanced.csv'
csv_fpath = '/home/tifr1/ms1/CLAM/Patient_wise_split_1k.csv'
# csv_fpath = '/home/tifr1/ms1/try/dataset_split.csv'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2023)

# display(pd.read_csv(csv_fpath).head(10)) # visualize data
loader_kwargs = {'batch_size': 1, 'num_workers': 2, 'pin_memory': False} # Batch size set to 1 due to variable bag sizes. Hard to collate.

class MILDataset(torch.utils.data.dataset.Dataset):
    r"""
    torch.utils.data.dataset.Dataset object that loads pre-extracted features per WSI from a CSV.

    Args:ABMIL_tut.ipynb
        feats_dirpath (str): Path to pre-extracted patch features (assumes that these features are saved as a *.pt object with it's corresponding slide_id as the filename)
        csv_fpath (str): Path to CSV file which contains: 1) Case ID, 2) Slide ID, 3) split information (train / val / test), and 4) label columns of interest for classification.
        which_split (str): Split that is used for subsetting the CSV (choices: ['train', 'val', 'test'])
        n_classes (int): Number of classes (default == 2 for LUAD vs LUSC subtyping)
    """
    def __init__(self, feats_dirpath=feats_dirpath, csv_fpath=csv_fpath, which_split='train', which_labelcol=which_labelcol):
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
    
    
train_dataset, val_dataset, test_dataset = [MILDataset(feats_dirpath, csv_fpath, which_split=split) for split in ['train', 'val', 'test']]

train_loader = torch.utils.data.DataLoader(train_dataset,shuffle=True, **loader_kwargs)
val_loader = torch.utils.data.DataLoader(val_dataset, shuffle=False, **loader_kwargs)
test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, **loader_kwargs)

def count_labels(dataset, label_col):
    counts = dataset.csv_split[label_col].value_counts().to_dict()
    return counts

# Count tumor vs normal in each split
train_counts = count_labels(train_dataset, which_labelcol)
val_counts = count_labels(val_dataset, which_labelcol)
test_counts = count_labels(test_dataset, which_labelcol)

print(f"Train samples: {len(train_dataset)} | Tumor: {train_counts.get(1, 0)} | Normal: {train_counts.get(0, 0)}")
print(f"Val samples: {len(val_dataset)} | Tumor: {val_counts.get(1, 0)} | Normal: {val_counts.get(0, 0)}")
print(f"Test samples: {len(test_dataset)} | Tumor: {test_counts.get(1, 0)} | Normal: {test_counts.get(0, 0)}")


from utils.io import read_patch_data
from utils.core import PseudoBag
from utils.func import seed_everything
seed_everything(42)

NUM_CLUSTER = 8 # the number of clusters
NUM_PSEB = 40 # the number of pseudo-bags
NUM_FT = 8 # # fine-tuning times
def pseudobag(bag_features):
    PB = PseudoBag(NUM_PSEB, NUM_CLUSTER, proto_method='mean', pheno_cut_method='quantile', iter_fine_tuning=NUM_FT)

    label_pseudo_bag_A = PB.divide(bag_features, ret_pseudo_bag=False).to(device)
    # print(f"[info] Bag A: it has {bag_features.shape[0]} instances.")
    # print(f"[info] Bag A: its first pseudo-bag has {(label_pseudo_bag_A == 0).sum()} instances.")
    # print(f"[info] Bag A: its second pseudo-bag has {(label_pseudo_bag_A == NUM_PSEB - 1).sum()} instances.")
    return label_pseudo_bag_A


PARAM_ALPHA = 1.0 # the parameter of Beta distribution
NUM_ITER = 10
def lam(PARAM_ALPHA = 1.0, NUM_ITER = 10):
    for i in range(NUM_ITER):
        lam = np.random.beta(PARAM_ALPHA, PARAM_ALPHA)
    return lam
# print(f"[info] current Mixup coefficient is {lam}")
lam = lam()
def lam_f(NUM_PSEB=NUM_PSEB,PARAM_ALPHA = 1.0):
    lam_temp = lam if lam != 1.0 else lam - 1e-5
    # print('lam_temp is ', lam_temp)
    lam_int  = int(lam_temp * (NUM_PSEB + 1))
    # print(f"[info] current Mixup coefficient (integer) is {lam_int}")
    return lam_int



def fetch_pseudo_bags(X, ind_X, n:int, n_parts:int):
    """
    X: bag features, usually with a shape of [N, d]
    ind_X: pseudo-bag indicator, usually with a shape of [N, ]
    n: pseudo-bag number, int
    n_parts: the pseudo-bag number to fetch, int
    """
    # print('shape is ',X.shape)
    
    if len(X.shape) > 2:
        X = X.squeeze(0)
    # print('x is ',X)
    # print("NUM_PSEB is :",n , "n is : ", )
    assert n_parts <= n, 'the pseudo-bag number to fetch is invalid.'
    # print("--------1")
    if n_parts == 0:
        print("--------none")
        return None
    # print("--------2")
    ind_fetched = torch.randperm(n)[:n_parts]
    X_fetched = torch.cat([X[ind_X == ind] for ind in ind_fetched], dim=0)

    return X_fetched


def mix_up(features1,label1,label2,features2,PROB_MIXUP = 0.98):
    label_pseudo_bag_A = pseudobag(features1) # Generate pseudo bags from n number of phenotype clusters
    label_pseudo_bag_B = pseudobag(features2)# Generate pseudo bags from n number of phenotype clusters

    bag_A = fetch_pseudo_bags(features1, label_pseudo_bag_A, NUM_PSEB, lam_f()) # fetch lam number of pseudo bags from total number of pseudo bags  
    bag_B = fetch_pseudo_bags(features2, label_pseudo_bag_B, NUM_PSEB, NUM_PSEB - lam_f()) # fetch (number of pseudo bags - lam) number of pseudo bags from total number of pseudo bags
    
    
    if np.random.rand() <= PROB_MIXUP and bag_A is not None and bag_B is not None: # our Random Mixup mechanism
        features = torch.cat([bag_A, bag_B], dim=0) # instance-axis concat
        mixup_ratio = lam_f() / NUM_PSEB
    elif np.random.rand() <= PROB_MIXUP and bag_A is None:
        features = bag_B
        mixup_ratio = 1.0
    elif np.random.rand() <= PROB_MIXUP and bag_B is None:
        features = bag_A
        mixup_ratio = 1.0
    elif bag_A is None and bag_B is None:
        features = features1
        mixup_ratio = 1.0
    else:
        features = bag_A
        mixup_ratio = 1.0

    return features,mixup_ratio

class AttentionTanhSigmoidGating(nn.Module):
    def __init__(self, D=64, L=64, dropout=0.25):
        super(AttentionTanhSigmoidGating, self).__init__()
        self.tanhV = nn.Sequential(*[nn.Linear(D, L), nn.Tanh(), nn.Dropout(dropout)])
        self.sigmU = nn.Sequential(*[nn.Linear(D, L), nn.Sigmoid(), nn.Dropout(dropout)])
        self.w = nn.Linear(L, 1)

    def forward(self, H):
        A_raw = self.w(self.tanhV(H).mul(self.sigmU(H))) # exponent term
        A_norm = F.softmax(A_raw, dim=0)                 # apply softmax to normalize weights to 1
        assert abs(A_norm.sum() - 1) < 1e-3              # Assert statement to check sum(A) ~= 1
        return A_norm


class ABMIL(nn.Module):
    def __init__(self, input_dim=feature_dimension, hidden_dim=64, dropout=0.25, n_classes=2):
        super(ABMIL, self).__init__()
        self.inst_level_fc = nn.Sequential(*[nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]) # Fully-Connected Layer, applied "instance-wise" to each embedding
        self.global_attn = AttentionTanhSigmoidGating(L=hidden_dim, D=hidden_dim)                              # Attention Function
        self.bag_level_classifier = nn.Linear(hidden_dim, n_classes)                                            # Bag-Level Classifier

    def forward(self, X: torch.randn(0, feature_dimension)):
        H_inst = self.inst_level_fc(X)         # 1. Process each feature embedding to be of size "hidden-dim"
        A_norm = self.global_attn(H_inst)      # 2. Get normalized attention scores for each embedding (s.t. sum(A_norm) ~= 1)
        z = torch.sum(A_norm * H_inst, dim=0)  # 3. Output of global attention pooling over the bag
        logits = self.bag_level_classifier(z).unsqueeze(dim=0)   # 4. Get un-normalized logits for classification task
        try:
            assert logits.shape == (1,2)
        except:
            print(f"Logit tensor shape is not formatted correctly. Should output [1 x 2] shape, but got {logits.shape} shape")
        return logits, A_norm
    
# model = ABMIL(input_dim=feature_dimension, hidden_dim=64).to(device)

# path = os.path.join('/drive2/shubham_tifr/abmil_l2.ckpt')

# model.eval()

def traineval_epoch(epoch, model, loader, optimizer=None,scheduler=None, loss_fn=nn.CrossEntropyLoss(), split='train', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose=1, print_every=300):
    model.train() if (split == 'train') else model.eval()       # turning on whether model should be used for training or evaluation
    total_loss, Y_probs, labels = 0.0, [], []                   # tracking loss + logits/labels for performance metrics
    x,t,n =0,0,0
    tum , norma = [], []
    results_list=[]
    bag_count = 0  # Count how many bags/slides are processed
    low_conf_correct_count = 0
    for batch_idx, (X_bag, X_bag1,slide_name) in enumerate(loader):
        # Since we assume batch size == 1, we want to prevent torch from collating our bag of patch features as [1 x M x D] torch tensors.
        # print(X_bag.shape)to(device)
        X_bag, label_bag = X_bag[0][0].to(device), X_bag[1].to(device)
        X_bag1,label1 = X_bag1[0][0].to(device), X_bag1[1].to(device)
        if label_bag == torch.LongTensor([0]).to(device) and label1 == torch.LongTensor([0]).to(device):
            label = torch.LongTensor([0]).to(device)
        else:
            label = torch.LongTensor([1]).to(device)
        # mix_bag = mix_up(X_bag, X_bag1)
        # print(f"[info] current Mixup coefficient is {lam}")  
        mix_bag,mixup_ratio  = mix_up(X_bag,label_bag,label1,X_bag1)
        # print(X_bag.shape)
        
        if mix_bag is None:
            mix_bag =X_bag
            print("*********---------***************")
            print("type of mix bag is x bag")
            print("*********---------***************")
            mixup_ratio  = 1.0       
        if (split == 'train'):
            #print(X_bag.shape)
            # print("**************type of mix bag is : ***********")
            # print("**************type of mix bag is : ***********")
            # print("type of mix bag is :",type(mix_bag))
            # print("**************type of mix bag is : ***********")
            # print("**************type of mix bag is : ***********") 
            logits, A_norm = model(mix_bag)
            # print(label.shape)
            loss = mixup_ratio * loss_fn(logits, label_bag) + (1 - mixup_ratio) * loss_fn(logits, label1)
            # def closure():
            #     optimizer.zero_grad()              # Clear previous gradients
            #     output = model(x)                  # Forward pass
            #     loss = loss        # Compute loss
            #     loss.backward()                    # Backward pass
            #     return loss
            x+=1
            if label == torch.LongTensor([1]).to(device):
                t+=1
            else:
                n+=1
            loss.backward(), optimizer.step(), optimizer.zero_grad()
        else:
            with torch.no_grad():
                logits, A_norm = model(X_bag)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                pred_label = np.argmax(probs)
                true_label = label_bag.item()
                
                # 👇 If correct but confidence is low
                if pred_label == true_label and probs[pred_label] < 0.7:
                    low_conf_correct_count += 1
                    print(f"{split} | [Low Conf] Slide: {slide_name} | Label: {true_label} | Pred: {pred_label} | Conf: {probs[pred_label]:.4f}")
                    
                # If wrong prediction, show confidence
                if pred_label != true_label:
                    print("---Wrong prediction---")
                    print(f"{split} | [Conf] Slide: {slide_name} | Label: {true_label} | Pred: {pred_label} | Conf: {probs[pred_label]:.4f}")
                    print("----------------------")
                results_list.append({
                    'split': split,
                    'slide_name': slide_name,
                    'true_label': true_label,
                    'pred_label': pred_label,
                    'confidence': probs[pred_label]
                })

            loss = loss_fn(logits, label_bag)
        total_loss += loss.item()
        Y_probs.append(torch.softmax(logits, dim=-1).cpu().detach().numpy())
        labels.append(label.cpu().detach().numpy())
        # if ((batch_idx + 1) % print_every == 0) and (verbose >= 2):
        #     print(f'Epoch {epoch}:\t Batch {batch_idx}\t Avg Loss: {total_loss / (batch_idx+1):.04f}\t Label: {label.item()}\t Bag Size: {X_bag.shape[0]}')

    # Final metrics
    Y_probs, labels = np.vstack(Y_probs), np.concatenate(labels)
    y_pred = Y_probs.argmax(axis=1)
    y_true = labels
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    log_dict = {
        f'{split} loss': total_loss / len(loader),
        f'{split} acc': sklearn.metrics.balanced_accuracy_score(y_true, y_pred),
        f'{split} auc': sklearn.metrics.roc_auc_score(y_true, Y_probs[:, 1]),
        f'{split} conf': cm,
        f'{split} FP': FP,
        f'{split} FN': FN,
        f'{split} count': bag_count,  # << ADD THIS
        f'{split} low_conf_count': low_conf_correct_count,
        f'{split} results': results_list if split != 'train' else None
    }

    if (verbose >= 1):
        print(f'### ({split.capitalize()} Summary) ###')
        print(f'Epoch {epoch}:\t' +
            f"{split} loss: {log_dict[f'{split} loss']:.04f}\t" +
            f"{split} acc: {log_dict[f'{split} acc']:.04f}\t" +
            f"{split} auc: {log_dict[f'{split} auc']:.04f}\t" +
            f"FP: {FP}\tFN: {FN}\tLow Conf: {low_conf_correct_count}\tCount: {bag_count}")
    return log_dict
    #         # print('type of logits ', logits)
    #         # print('type of label ', label)
    #         with torch.no_grad(): logits, A_norm = model(X_bag)
    #         # print('type of logits ', logits)
    #         # print('type of label ', label_bag)
    #         # loss = loss_fn(logits, label)
    #         # loss = mixup_ratio * loss_fn(logits, label_bag) + (1 - mixup_ratio) * loss_fn(logits, label1)
    #         loss = loss_fn(logits,label_bag)
    #     # print("x is ---------",x)
    #     # print("Number of tumor : ",t)
    #     # print("Number of normal :",n)
    #     tum.append(t)
    #     norma.append(n)
    #     # Track total loss, logits, and current progress
    #     total_loss += loss.item()
    #     Y_probs.append(torch.softmax(logits, dim=-1).cpu().detach().numpy())
    #     labels.append(label.cpu().detach().numpy())
    #     if ((batch_idx + 1) % print_every == 0) and (verbose >= 2):
    #         print(f'Epoch {epoch}:\t Batch {batch_idx}\t Avg Loss: {total_loss / (batch_idx+1):.04f}\t Label: {label.item()}\t Bag Size: {X_bag.shape[0]}')

    # df1 = pd.DataFrame([tum, norma])
    # df1.to_csv('num_tumor_normal.csv', index=False, header=False)
    # # Compute balanced accuracy and AUC-ROC from saved logits / labels
    # Y_probs, labels = np.vstack(Y_probs), np.concatenate(labels)
    # log_dict = {f'{split} loss': total_loss/len(loader),
    #             f'{split} acc': sklearn.metrics.balanced_accuracy_score(labels, Y_probs.argmax(axis=1)),
    #             f'{split} auc': sklearn.metrics.roc_auc_score(labels, Y_probs[:, 1])}

    # # Print out end-of-epoch information
    # if (verbose >= 1):
    #     print(f'### ({split.capitalize()} Summary) ###')
    #     print(f'Epoch {epoch}:\t' + f'\t'.join([f'{k.capitalize().rjust(10)}: {log_dict[k]:.04f}' for k,v in log_dict.items()]))
    # return log_dict


# Get model, optimizer, and loss function 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
model = ABMIL(input_dim=feature_dimension, hidden_dim=64).to(device)
# path = os.path.join('abmil_rand_mix_sc_lr_0.5.ckpt')
# path = os.path.join('/drive2/shubham_tifr/ouputs/data_balanced_lr_e-5/abmil_sc_lr_0.5.ckpt')
# path = os.path.join('/drive2/shubham_tifr/abmil_l5_sched_lr.ckpt')
# path = os.path.join('/home/tifr1/ms1/ABMIL_N0/output/adamw_ylim/13july/abmil__13_july.ckpt')
# path = os.path.join('/home/tifr1/ms1/ABMIL_N0/output/adamw_ylim/24july/abmil__24_pse_july_mor.ckpt')
# model.load_state_dict(torch.load(path))
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
# optimizer = torch.optim.LBFGS(model.parameters(), lr=1e-5, max_iter = 20)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.05, patience=5, verbose=True)

# Example: if you have 3x more class 0 than class 1
# pos_weight = torch.tensor([3.0])  # scalar tensor

# loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_fn = nn.CrossEntropyLoss()
# loss_fn1 = nn.CrossEntropyLoss(label_smoothing=0.3)
# Set-up train-validation loop and early stopping

num_epochs, min_early_stopping, patience, counter = 50,35,10,0
lowest_val_loss, best_model = np.inf, None
all_train_logs, all_val_logs, all_test_logs = [], [], [] # TODO: do something with train_log / val_log every epoch to help visualize performance curves?
# print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | Test samples: {len(test_dataset)}")
for epoch in range(num_epochs):
    # 🔢 Print how many samples in each split at the start of every epoch
    print(f"\n--- Epoch {epoch} ---")
    

    # 🔁 Run training and validation
    train_log = traineval_epoch(epoch, model, train_loader, optimizer=optimizer, split='train', device=device, verbose=2, print_every=100)
    val_log = traineval_epoch(epoch, model, val_loader, optimizer=None, split='val', device=device, verbose=1)

    # print("Val low-confidence correct predictions (<0.7):", val_log['val low_conf_count'])

    # 📦 Store logs
    all_train_logs.append(train_log)
    all_val_logs.append(val_log)

    # 🎯 Early stopping logic
    val_loss = val_log['val loss']
    if epoch > min_early_stopping:
        if val_loss < lowest_val_loss:
            print(f"Resetting early-stopping counter: {lowest_val_loss:.04f} -> {val_loss:.04f}")
            lowest_val_loss, counter, best_model = val_loss, 0, copy.deepcopy(model)
        else:
            counter += 1
            print(f"Early-stopping counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered!")
            break

    # 🔄 Learning rate scheduler
    scheduler.step(val_loss)
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"Epoch {epoch} - LR of group {i}: {param_group['lr']}")

    print()
# num_epochs, min_early_stopping, patience, counter = 30,15,5,0
# lowest_val_loss, best_model = np.inf, None
# all_train_logs, all_val_logs, all_test_logs = [], [], [] # TODO: do something with train_log / val_log every epoch to help visualize performance curves?
# for epoch in range(num_epochs):
#     train_log = traineval_epoch(epoch, model, train_loader, optimizer=optimizer, split='train', device=device, verbose=2, print_every=100)
#     val_log = traineval_epoch(epoch, model, val_loader, optimizer=None, split='val', device=device, verbose=1)
#     # test_log1 = traineval_epoch(epoch, model, test_loader, optimizer=None, split='test', device=device, verbose=1)
#     all_train_logs.append(train_log)
#     all_val_logs.append(val_log)
#     # all_test_logs.append(test_log1)
#     val_loss = val_log['val loss']
#     scheduler.step(val_loss)
#     print("learning rate is : ",optimizer.param_groups[0]["lr"],"*****************************************************")
#     # Early stopping: If validation loss does not go down for <patience> epochs after <min_early_stopping> epochs, stop model training early
#     if (epoch > min_early_stopping):
#         if (val_loss < lowest_val_loss):
#             print(f'Resetting early-stopping counter: {lowest_val_loss:.04f} -> {val_loss:.04f}...')
#             lowest_val_loss, counter, best_model = val_loss, 0, copy.deepcopy(model)
#         else:
#             print(f'Early-stopping counter updating: {counter}/{patience} -> {counter+1}/{patience}...')
#             counter += 1

#     if counter >= patience: break
#     print()

# Report best model (lowest validation loss) on test split
best_model = model if (best_model is None) else best_model
test_log = traineval_epoch(epoch, best_model, test_loader, optimizer=None, split='test', device=device, verbose=1)


torch.save(best_model.state_dict(), 'abmil__2_pse_sep.ckpt')


# 100 mix bags 30 psedo bags

import matplotlib.pyplot as plt
import pickle
# Extract losses and accuracies
train_losses = [log['train loss'] for log in all_train_logs]
val_losses = [log['val loss'] for log in all_val_logs]
train_accs = [log['train acc'] for log in all_train_logs]
val_accs = [log['val acc'] for log in all_val_logs]
# test_losses = [log['test loss'] for log in all_test_logs]
# test_accs = [log['test acc'] for log in all_test_logs]

with open('train_losses_2sep.pkl', 'wb') as f:
    pickle.dump(train_losses, f)
with open('val_losses_2sep.pkl', 'wb') as f:
    pickle.dump(val_losses, f)
# with open('test_losses_25_pse_eve_july.pkl', 'wb') as f:
#     pickle.dump(test_losses, f)
    
    
# Plotting
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Val Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
# plt.ylim(0, 0.5)
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Accuracy', marker='o')
plt.plot(val_accs, label='Val Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')
plt.title('Accuracy over Epochs')
# plt.ylim(0, 0.5)
plt.legend()

plt.tight_layout()
plt.savefig('abmil__lr_2sep_pse.png')
# plt.show()

# Plot Loss
plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Train Loss', marker='o')
# plt.plot(val_losses, label='Val Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.ylim(0, 0.6)
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
# plt.plot(train_accs, label='Train Accuracy', marker='o')
# plt.plot(val_accs, label='Val Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')
plt.title('Accuracy over Epochs')
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig('abmil__lr_2sep_pse_0_5.png')

# Plot Loss
plt.subplot(1, 2, 1)
# plt.plot(train_losses, label='Train Loss', marker='o')
# plt.plot(val_losses, label='Val Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.ylim(0, 0.8)
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
# plt.plot(train_accs, label='Train Accuracy', marker='o')
# plt.plot(val_accs, label='Val Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Balanced Accuracy')
plt.title('Accuracy over Epochs')
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.savefig('abmil__lr_2sep_pse_0.6.png')
# plt.subplot(1, 2, 1)
# # plt.plot(train_losses, label='Train Loss', marker='o')
# # plt.plot(val_losses, label='Val Loss', marker='o')abmil__13_july
# plt.plot(test_losses, label='Test Loss', marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss over Epochs')
# # plt.ylim(0, 1)
# plt.legend()

# # Plot Accuracy
# plt.subplot(1, 2, 2)
# # plt.plot(train_accs, label='Train Accuracy', marker='o')
# # plt.plot(val_accs, label='Val Accuracy', marker='o')
# plt.plot(test_accs, label='Test Accuracy', marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Balanced Accuracy')
# plt.title('Accuracy over Epochs')
# # plt.ylim(0, 1)
# plt.legend()

# plt.tight_layout()
# plt.savefig('abmil__lr_26aug_pse_test.png')

# plt.subplot(1, 2, 1)
# # plt.plot(train_losses, label='Train Loss', marker='o')
# # plt.plot(val_losses, label='Val Loss', marker='o')abmil__13_july
# plt.plot(test_losses, label='Test Loss', marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Loss over Epochs')
# # plt.ylim(0, 1)
# plt.legend()

# # Plot Accuracy
# plt.subplot(1, 2, 2)
# # plt.plot(train_accs, label='Train Accuracy', marker='o')
# # plt.plot(val_accs, label='Val Accuracy', marker='o')
# plt.plot(test_accs, label='Test Accuracy', marker='o')
# plt.xlabel('Epoch')
# plt.ylabel('Balanced Accuracy')
# plt.title('Accuracy over Epochs')
# # plt.ylim(0, 1)
# plt.legend()

# plt.tight_layout()
# plt.savefig('abmil__lr_26aug_pse_test.png')