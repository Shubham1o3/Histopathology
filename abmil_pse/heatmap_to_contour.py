import os
import json
import argparse
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import h5py
from PIL import Image
from openslide import OpenSlide
import matplotlib.pyplot as plt
from models.ABMIL import ABMIL   
from utils.heatmap import HeatmapGenerator   


def heatmap_to_binary_mask(heatmap_img, slide_id, scale_x=1.0, scale_y=1.0):
    """
    Converts heatmap to binary mask, .geojson, and .groovy 
    """
    # Prepare directories
    base_dir = os.path.join("qupath_files", slide_id)
    os.makedirs(base_dir, exist_ok=True)

    # Prepare output paths
    binary_mask_path = os.path.join(base_dir, f"{slide_id}_binary_mask.tif")
    geojson_path = os.path.join(base_dir, f"{slide_id}.geojson")
    groovy_path = os.path.join(base_dir, f"{slide_id}.groovy")

    # Convert to array and HSV
    heatmap_np = np.array(heatmap_img.convert("RGB"))
    hsv = cv2.cvtColor(heatmap_np, cv2.COLOR_RGB2HSV)

    # Red detection
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Contours
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    binary_mask = np.zeros_like(red_mask)
    if contours:
        cv2.drawContours(binary_mask, contours, -1, color=255, thickness=cv2.FILLED)

    # Save binary mask
    Image.fromarray(binary_mask).save(binary_mask_path)

    # Save GeoJSON
    geojson = {"type": "FeatureCollection", "features": []}
    for contour in contours:
        pts = contour.squeeze()
        if len(pts.shape) != 2:
            continue
        polygon = [[float(x * scale_x), float(y * scale_y)] for x, y in pts.tolist()]
        if polygon[0] != polygon[-1]:
            polygon.append(polygon[0])
        feature = {
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [polygon]},
            "properties": {}
        }
        geojson["features"].append(feature)

    with open(geojson_path, "w") as f:
        json.dump(geojson, f, indent=2)

    # Save Groovy
    groovy_lines = [
        "import qupath.lib.objects.PathAnnotationObject",
        "import qupath.lib.objects.classes.PathClass",
        "import qupath.lib.roi.ROIs",
        "import qupath.lib.geom.Point2",
        "import qupath.lib.regions.ImagePlane",
        "",
        "def annotationClass = PathClass.fromString('Prediction')",
        "def plane = ImagePlane.getDefaultPlane()",
        "int added = 0",
        ""
    ]
    for i, feature in enumerate(geojson["features"]):
        coords = feature["geometry"]["coordinates"][0]
        groovy_lines.append(f"// --- Polygon {i} ---")
        groovy_lines.append(f"def points_{i} = [")
        for x, y in coords:
            groovy_lines.append(f"    new Point2({x:.2f}, {y:.2f}),")
        groovy_lines.append("]")
        groovy_lines.append(f"def roi_{i} = ROIs.createPolygonROI(points_{i}, plane)")
        groovy_lines.append(f"def annotation_{i} = new PathAnnotationObject(roi_{i}, annotationClass)")
        groovy_lines.append(f"addObject(annotation_{i})")
        groovy_lines.append("added++\n")

    groovy_lines.append('print "Imported " + added + " annotation(s)"')

    with open(groovy_path, "w") as g:
        g.write("\n".join(groovy_lines))

    print(f"Saved binary mask, geojson, and groovy in: {base_dir}")
    return binary_mask


def main(args):
    device = torch.device(args.device)

    # Load features + coords
    features = torch.load(args.feats_path).to(device)
    with h5py.File(args.h5_path, "r") as f:
        coords = f["coords"][:]

    # Load WSI
    wsi = OpenSlide(args.slide_path)
    width, height = wsi.dimensions

    # Load model
    model = ABMIL(input_dim=args.feature_dim).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # Run inference
    logits, attention = model(features, return_raw_attention=True)
    attention = attention.squeeze().detach().cpu().numpy()

    # Draw heatmap
    gen = HeatmapGenerator(args.model, args.feats_path, args.h5_path, args.slide_path,
                            feature_dim=args.feature_dim, device=device)
    heatmap = gen.draw_heatmap(attention, easy_plot=True)

    # Convert to binary mask + export
    binary_mask = heatmap_to_binary_mask(
        heatmap,
        slide_id=args.slide,
        scale_x=width / heatmap.size[0],
        scale_y=height / heatmap.size[1]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", type=str, required=True, help="Slide ID (without extension)")
    parser.add_argument("--slide_path", type=str, required=True, help="Path to whole slide image (.svs)")
    parser.add_argument("--h5_path", type=str, required=True, help="Path to slide h5 coords file")
    parser.add_argument("--feats_path", type=str, required=True, help="Path to slide features (.pt)")
    parser.add_argument("--model", type=str, required=True, help="Path to trained ABMIL checkpoint")
    parser.add_argument("--feature_dim", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    main(args)


# python heatmap_to_contour.py \
#   --slide CAIB-T00004280OC01R01P0404HE \
#   --slide_path /drive2/N0DATA/CAIB-T00004280OC01R01P0404HE.svs \
#   --h5_path /home/tifr1/ms1/CLAM/FEATURES_DIRECTORY4000/h5_files/CAIB-T00004280OC01R01P0404HE.h5 \
#   --feats_path /home/tifr1/ms1/CLAM/FEATURES_DIRECTORY4000/pt_files/CAIB-T00004280OC01R01P0404HE.pt \
#   --model outputs/abmil_model.ckpt
