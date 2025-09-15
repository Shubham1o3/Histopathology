import argparse
import os
from utils.heatmap import HeatmapGenerator
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (.ckpt)")
    parser.add_argument("--feats_path", type=str, required=True, help="Path to slide features (.pt file)")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to slide h5 coords file")
    parser.add_argument("--slide_path", type=str, required=True, help="Path to whole slide image (.svs)")
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_heatmap", type=str, default="heatmap_output.png",
                        help="Path to save the generated heatmap image")
    args = parser.parse_args()

    # Run
    gen = HeatmapGenerator(args.model, args.feats_path, args.h5_path, args.slide_path,
                        feature_dim=args.feature_dim, device=args.device)
    predicted, probs, attention = gen.run_inference()
    print(f"Prediction: {predicted}, Probabilities: {probs}")

    # Generate heatmap
    heatmap = gen.draw_heatmap(attention, easy_plot=True)
    # Ensure directory exists
    save_dir = os.path.dirname(args.save_heatmap)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        
    # Save instead of show
    heatmap.save(args.save_heatmap)
    print(f" Heatmap saved to {args.save_heatmap}")
    
# python heatmap.py \
#   --model /home/tifr1/ms1/folder/abmil-plain/trial1/abmil_untitled_11aug.ckpt \
#   --feats_path /home/tifr1/ms1/CLAM/FEATURES_DIRECTORY4000/pt_files/CAIB-T00004251OC01R01P0102HE.pt \
#   --h5_path /home/tifr1/ms1/CLAM/FEATURES_DIRECTORY4000/h5_files/CAIB-T00004251OC01R01P0102HE.h5 \
#   --slide_path /drive2/N0DATA/CAIB-T00004251OC01R01P0102HE.svs \
#   --save_heatmap ./outputs/CAIB-T00004251OC01R01P0102HE_heatmap.png