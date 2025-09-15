from utils.cm import MILInference
import argparse
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABMIL Inference Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model checkpoint (.ckpt)")
    parser.add_argument("--csv_fpath", type=str, required=True, help="Path to CSV file with split info")
    parser.add_argument("--feats_path", type=str, required=True, help="Directory containing .pt feature files")
    parser.add_argument("--input_dim", type=int, default=1024, help="Input feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension in ABMIL")
    parser.add_argument("--labelcol", type=str, default="Binary_label", help="Column name for labels")
    args = parser.parse_args()

    # Auto device selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    inference = MILInference(
        model_path=args.model_path,
        csv_fpath=args.csv_fpath,
        feats_path=args.feats_path,
        device=device,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        labelcol=args.labelcol
    )

    results_df = inference.run()
    results_df.to_csv("inference_results.csv", index=False)
    print("Inference complete. Results saved to inference_results.csv")
    
    
# python confusion_matrix.py \
#   --model_path /home/tifr1/ms1/folder/abmil-plain/trail2/12aug/abmil_untitled_12aug.ckpt \
#   --csv_fpath /home/tifr1/ms1/CLAM/Patient_wise_split_1k.csv \
#   --feats_path /home/tifr1/ms1/CLAM/FEATURES_DIRECTORY4000/pt_files

