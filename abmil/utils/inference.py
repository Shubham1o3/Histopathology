from models.ABMIL import ABMIL
import os
import pandas as pd
import argparse
import numpy as np 

def run_inference(model_path, slide_id, feats_dirpath, csv_path, label_col, feature_dim=1024, device="cpu"):
    # Load CSV and label
    csv = pd.read_csv(csv_path)
    label = csv[csv['slide_id'] == slide_id][label_col].values[0]

    # Load features
    feat_path = os.path.join(feats_dirpath, slide_id + ".pt")
    features = torch.load(feat_path).to(device)

    # Load model
    model = ABMIL(input_dim=feature_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Forward pass
    logits, attention = model(features, return_raw_attention=True)
    logits = logits.squeeze()
    attention = attention.squeeze().detach().cpu().numpy()
    probs = F.softmax(logits, dim=0)

    # Prediction
    predicted = torch.argmax(probs).item()

    # Print results
    print(f"Slide: {slide_id}")
    print(f"Label: {label}")
    print(f"Features Shape: {features.shape}")
    print(f"Attention Shape: {attention.shape}, Min: {np.min(attention):.4f}, Max: {np.max(attention):.4f}")
    print(f"Logits: {logits.detach().cpu().numpy()}")
    print(f"Prediction Class: {predicted}, Probabilities: {probs.detach().cpu().numpy()}")

    return predicted, probs.detach().cpu().numpy(), attention
