import torch
import pandas as pd 
from models.ABMIL import *
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

class MILInference:
    def __init__(self, model_path, csv_fpath, feats_path, device="cuda", input_dim=1024, hidden_dim=64, labelcol="Binary_label"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = ABMIL(input_dim=input_dim, hidden_dim=hidden_dim).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.csv = pd.read_csv(csv_fpath)
        self.csv_split_test = self.csv[self.csv['split'] == 'test']
        self.feats_path = feats_path
        self.labelcol = labelcol

    def run(self):
        predicted_labels = []
        true_labels = []

        for i in range(len(self.csv_split_test)):
            slide_id = self.csv_split_test.iloc[i]['slide_id']
            label = self.csv_split_test.iloc[i][self.labelcol]
            true_labels.append(label)

            # Load features
            features = torch.load(os.path.join(self.feats_path, slide_id + '.pt')).to(self.device)

            # Run model
            logits, attention = self.model(features, return_raw_attention=True)
            logits = logits.squeeze()

            # Predict
            probs = F.softmax(logits, dim=0)
            predicted = torch.argmax(probs).item()
            predicted_labels.append(predicted)

            print(f"Slide: {slide_id}, True: {label}, Pred: {predicted}, Probs: {probs.tolist()}")

        # Save predictions in DataFrame
        self.csv_split_test["Predicted_labels"] = predicted_labels

        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        print("\nConfusion Matrix:\n", cm)

        # Plot confusion matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal', 'Tumor'],
                    yticklabels=['Normal', 'Tumor'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.savefig('confusion_matrix.png')

        return self.csv_split_test