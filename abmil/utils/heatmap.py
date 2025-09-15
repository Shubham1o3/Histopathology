from models.ABMIL import *
import numpy as np
import h5py
from openslide import OpenSlide
from PIL import Image
import matplotlib.pyplot as plt
import torch

class HeatmapGenerator:
    def __init__(self, model_path, feats_path, h5_path, slide_path,
                feature_dim=1024, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model_path = model_path
        self.feats_path = feats_path
        self.h5_path = h5_path
        self.slide_path = slide_path
        self.device = device

        # Load features
        self.features = torch.load(self.feats_path).to(device)

        # Load coords from h5
        with h5py.File(self.h5_path, "r") as f:
            self.coords = f["coords"][:]

        # Load WSI
        self.wsi = OpenSlide(self.slide_path)

        # Load model
        self.model = ABMIL(input_dim=feature_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def run_inference(self):
        logits, attention = self.model(self.features, return_raw_attention=True)
        logits = logits.squeeze()
        attention = attention.squeeze().detach().cpu().numpy()
        probs = F.softmax(logits, dim=0)
        predicted = torch.argmax(probs).item()
        return predicted, probs.detach().cpu().numpy(), attention

    def draw_heatmap(self, scores, vis_level=0, patch_size=(256, 256),
                    cmap="coolwarm", custom_downsample=4, easy_plot=True, both=False):
        D = custom_downsample
        region_size = self.wsi.level_dimensions[vis_level]
        w, h = region_size
        w, h = w // D, h // D
        patch_size = np.ceil(np.array(patch_size) / D).astype(int)
        coords = np.ceil(self.coords / D).astype(int)

        print(f"Creating heatmap: w={w}, h={h}, total patches={len(coords)}")

        overlay = np.zeros((h, w), dtype=float)
        img = np.full((h, w, 3), (255, 255, 255), dtype=np.uint8)

        cmap = plt.get_cmap(cmap)
        norm = plt.Normalize(scores.min(), scores.max())

        for idx, coord in enumerate(coords):
            score = scores[idx].item()
            overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] += score
            raw_block = overlay[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]]
            color_block = (cmap(norm(raw_block)) * 255)[:, :, :3].astype(np.uint8)
            img[coord[1]:coord[1]+patch_size[1], coord[0]:coord[0]+patch_size[0]] = color_block.copy()
        del overlay
        heatmap_img = Image.fromarray(img)

        if both:
            org_img = self.wsi.read_region((0, 0), vis_level, region_size).convert("RGB")
            org_img = org_img.resize((int(w / D), int(h / D)), resample=Image.BILINEAR)
            return heatmap_img.resize((int(w / D), int(h / D))), org_img

        if easy_plot:
            return heatmap_img.resize((int(w / D), int(h / D)))

        return heatmap_img
