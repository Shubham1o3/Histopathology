import sklearn.metrics
import numpy as np 
import torch
import torch.nn as nn


def traineval_epoch(epoch, model, loader, optimizer=None,scheduler=None, loss_fn=nn.CrossEntropyLoss(), split='train', device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), verbose=1, print_every=300):
    model.train() if (split == 'train') else model.eval()       
    total_loss, Y_probs, labels = 0.0, [], []                   
    x,t,n =0,0,0
    tum , norma = [], []
    results_list=[]
    bag_count = 0  # Count how many bags/slides are processed
    low_conf_correct_count = 0
    for batch_idx, (X_bag, X_bag1,slide_name) in enumerate(loader):
        X_bag, label_bag = X_bag[0][0].to(device), X_bag[1].to(device)
        X_bag1,label1 = X_bag1[0][0].to(device), X_bag1[1].to(device)
        if label_bag == torch.LongTensor([0]).to(device) and label1 == torch.LongTensor([0]).to(device):
            label = torch.LongTensor([0]).to(device)
        else:
            label = torch.LongTensor([1]).to(device)
        from utils.pseudo import PseudoMixup
        pm=PseudoMixup(num_cluster=8) 
        mix_bag,mixup_ratio  = pm.mix_up(X_bag,label_bag,label1,X_bag1)
        
        if mix_bag is None:
            mix_bag =X_bag
            print("*********---------***************")
            print("Mix bag is not working")
            print("*********---------***************")
            mixup_ratio  = 1.0       
        if (split == 'train'):
            logits, A_norm = model(mix_bag)
            loss = mixup_ratio * loss_fn(logits, label_bag) + (1 - mixup_ratio) * loss_fn(logits, label1)
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

    # metrics
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