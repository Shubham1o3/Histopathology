import torch
from utils.inference import run_inference
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (.ckpt)")
    parser.add_argument("--slide", type=str, required=True, help="Slide ID (e.g. CAIB-XXXX)")
    parser.add_argument("--feats_dirpath", type=str, required=True, help="Path to folder with .pt feature files")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file with slide metadata")
    parser.add_argument("--label_col", type=str, default="Binary_label", help="Column name for labels in CSV")
    parser.add_argument("--feature_dim", type=int, default=1024, help="Feature dimension")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    run_inference(
        model_path=args.model,
        slide_id=args.slide,
        feats_dirpath=args.feats_dirpath,
        csv_path=args.csv_path,
        label_col=args.label_col,
        feature_dim=args.feature_dim,
        device=args.device,
    )
    
# python inference.py \
#   --model /home/tifr1/ms1/folder/abmil-plain/trial1/abmil_untitled_11aug.ckpt \
#   --slide CAIB-T00004251OC01R01P0102HE \
#   --feats_dirpath /home/tifr1/ms1/CLAM/FEATURES_DIRECTORY4000/pt_files \
#   --csv_path /home/tifr1/ms1/CLAM/Patient_wise_split_1k.csv \
#   --label_col 'Binary_label'
