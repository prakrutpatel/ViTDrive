import torch
import torch.backends.cudnn as cudnn
import yaml
import os
import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import torch.nn.functional as F
from tqdm import tqdm
import torchvision.transforms as T
from PIL import Image # Use PIL for robust image loading

# Assuming ViTDrive is in PYTHONPATH or accessible
from RT_DETR.rtdetrv2_pytorch.src.core import YAMLConfig
from model.model import TwinLiteNetPlus
from utils import generate_rtdetr_heatmap # Only need heatmap generation from utils

# Define colors for visualization (BGR format for OpenCV)
DA_COLOR = [0, 255, 0]  # Green for Driving Area
LL_COLOR = [255, 0, 0]  # Blue for Lane Lines
ALPHA = 0.4 # Transparency for segmentation overlay
HEATMAP_ALPHA = 0.5 # Transparency for heatmap overlay

def load_twinlitenet_model(args, device):
    """Loads the TwinLiteNetPlus model from a state_dict file."""
    print("=> Initializing TwinLiteNetPlus model...")
    # We need args.config for model structure, create a dummy object if needed
    # or ensure config is passed correctly
    model_args = lambda: None # Simple namespace object
    setattr(model_args, 'config', args.config)
    model = TwinLiteNetPlus(model_args).to(device)

    print(f"=> Loading TwinLiteNetPlus weights: {args.weights}")
    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"ERROR: TwinLiteNetPlus weights file not found at '{args.weights}'")

    try:
        # Load the state dictionary directly (standalone model state)
        # Use weights_only=True for safety as we only need the state_dict
        state_dict = torch.load(args.weights, map_location=device, weights_only=True)

        # Handle potential nesting or 'module.' prefix if saved differently
        if 'state_dict' in state_dict: state_dict = state_dict['state_dict']
        if 'ema_state_dict' in state_dict: state_dict = state_dict['ema_state_dict'] # Prefer EMA if nested
        if 'model' in state_dict: state_dict = state_dict['model']

        # Remove 'module.' prefix if it exists (from DDP saving)
        if all(k.startswith('module.') for k in state_dict.keys()):
            print("   Removing 'module.' prefix from weights.")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        print("=> TwinLiteNetPlus weights loaded successfully.")
        model.eval()
        return model

    except Exception as e:
        print(f"ERROR: Failed to load TwinLiteNetPlus weights: {e}")
        raise e

def load_rtdetr_model(args, device):
    """Loads the RT-DETR model."""
    print("=> Loading RT-DETR model for heatmap generation...")
    try:
        rtdetr_cfg = YAMLConfig(args.rtdetr_config)
        # Load on CPU first
        rtdetr_checkpoint = torch.load(args.rtdetr_resume, map_location='cpu', weights_only=True)

        # Standard practice is to load the state_dict directly if it's just weights
        # Adjust logic if your RT-DETR checkpoint has nesting like 'ema' or 'model'
        if isinstance(rtdetr_checkpoint, dict):
            if 'ema' in rtdetr_checkpoint and 'module' in rtdetr_checkpoint['ema']:
                 rtdetr_state = rtdetr_checkpoint['ema']['module']
                 print("   Loading RT-DETR EMA weights.")
            elif 'model' in rtdetr_checkpoint:
                 rtdetr_state = rtdetr_checkpoint['model']
                 print("   Loading RT-DETR standard weights ('model').")
            else:
                 # Assume the dict itself is the state_dict if common keys aren't found
                 rtdetr_state = rtdetr_checkpoint
                 print("   Loading RT-DETR weights (assuming dict is state_dict).")
        else:
            # If it's not a dict, assume it's the state_dict directly
            rtdetr_state = rtdetr_checkpoint
            print("   Loading RT-DETR weights (assuming file is state_dict).")


        rtdetr_model = rtdetr_cfg.model
        rtdetr_model.load_state_dict(rtdetr_state)
        print("=> RT-DETR model loaded successfully.")

        rtdetr_model = rtdetr_model.to(device)
        rtdetr_model.eval()

        print("=> Freezing RT-DETR parameters...")
        for param in rtdetr_model.parameters():
            param.requires_grad = False
        print("=> RT-DETR parameters frozen.")
        return rtdetr_model

    except FileNotFoundError:
        print(f"ERROR: RT-DETR config or weights file not found.")
        print(f"  Config: '{args.rtdetr_config}'")
        print(f"  Weights: '{args.rtdetr_resume}'")
        raise
    except Exception as e:
        print(f"ERROR: Failed to load RT-DETR model: {e}")
        raise e

def inference(args):
    """
    Perform inference on a directory of images.
    """
    # --- Setup ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_heatmap_dir = output_dir / "heatmap_overlay"
    output_seg_dir = output_dir / "segmentation_overlay"
    output_heatmap_dir.mkdir(exist_ok=True)
    output_seg_dir.mkdir(exist_ok=True)

    # --- Load Models ---
    try:
        rtdetr_model = load_rtdetr_model(args, device)
        twinlitenet_model = load_twinlitenet_model(args, device)
    except Exception as e:
        print(f"Model loading failed: {e}. Exiting.")
        return

    if args.half and device.type == 'cuda':
        print("Using half precision (FP16).")
        twinlitenet_model = twinlitenet_model.half()
        # RT-DETR heatmap generation often expects float, keep it float unless tested otherwise
        # rtdetr_model = rtdetr_model.half() # Optional, maybe keep float
    dtype = torch.half if args.half and device.type == 'cuda' else torch.float

    # --- Image Transformations ---
    img_h, img_w = args.img_size
    transform = T.Compose([
        T.Resize((img_h, img_w)),
        T.ToTensor(), # Converts to [0, 1] range automatically
        # Add normalization if the model was trained with it
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # --- Find Input Images ---
    image_files = []
    supported_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    print(f"Searching for images in: {args.input_dir}")
    for ext in supported_exts:
        image_files.extend(Path(args.input_dir).glob(f'*{ext}'))
    print(f"Found {len(image_files)} images.")

    # --- Inference Loop ---
    for img_path in tqdm(image_files, desc="Processing Images"):
        try:
            # Load image using PIL (handles various formats, converts to RGB)
            img_pil = Image.open(img_path).convert('RGB')
            original_w, original_h = img_pil.size

            # Preprocess for Model Input
            input_tensor = transform(img_pil).unsqueeze(0).to(device).to(dtype) # Add batch dim

            # --- Generate Heatmap ---
            # Heatmap generation expects float input
            heatmap_tensor = generate_rtdetr_heatmap(rtdetr_model, input_tensor.float(),
                                                    target_size=(img_h, img_w), device=device)
            heatmap_tensor = heatmap_tensor.to(dtype) # Convert back to desired dtype if needed

            # --- Run Inference ---
            with torch.no_grad():
                output_da, output_ll = twinlitenet_model(input_tensor, heatmap_tensor)

            # --- Post-process Segmentation ---
            # Resize logits to *original* image size for visualization
            out_da_resized = F.interpolate(output_da, size=(original_h, original_w), mode='bilinear', align_corners=False)
            out_ll_resized = F.interpolate(output_ll, size=(original_h, original_w), mode='bilinear', align_corners=False)

            # Get prediction class maps
            da_predict = torch.argmax(out_da_resized, dim=1).squeeze().cpu().numpy().astype(np.uint8)
            ll_predict = torch.argmax(out_ll_resized, dim=1).squeeze().cpu().numpy().astype(np.uint8)

            # --- Visualization ---
            # Load original image again with OpenCV for BGR format needed by cv2 functions
            img_cv2 = cv2.imread(str(img_path))
            # Resize if original image was huge, otherwise use original size
            # img_cv2 = cv2.resize(img_cv2, (original_w, original_h)) # Ensure size consistency

            # 1. Heatmap Overlay
            heatmap_vis = heatmap_tensor.squeeze().cpu().numpy() # H, W, Range [0, 1]
            heatmap_vis = cv2.resize(heatmap_vis, (original_w, original_h)) # Resize to original
            heatmap_vis = (heatmap_vis * 255).astype(np.uint8) # Scale to 0-255
            heatmap_colored = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET) # Apply colormap
            heatmap_overlay = cv2.addWeighted(img_cv2, 1 - HEATMAP_ALPHA, heatmap_colored, HEATMAP_ALPHA, 0)

            # 2. Segmentation Overlay
            seg_overlay = img_cv2.copy()
            da_mask = da_predict == 1
            ll_mask = ll_predict == 1
            seg_overlay[da_mask] = DA_COLOR
            seg_overlay[ll_mask] = LL_COLOR # Lane lines drawn over driving area if they overlap
            # Blend original image with the colored masks
            final_seg_overlay = cv2.addWeighted(img_cv2, 1 - ALPHA, seg_overlay, ALPHA, 0)


            # --- Save Outputs ---
            base_filename = img_path.stem
            heatmap_out_path = output_heatmap_dir / f"{base_filename}_heatmap.png"
            seg_out_path = output_seg_dir / f"{base_filename}_seg.png"

            cv2.imwrite(str(heatmap_out_path), heatmap_overlay)
            cv2.imwrite(str(seg_out_path), final_seg_overlay)

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging


if __name__ == '__main__':
    parser = ArgumentParser()
    # --- Paths ---
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save output visualization images.')
    parser.add_argument('--weights', type=str, required=True, help='Path to the trained TwinLiteNetPlus model weights (.pth file, standalone state_dict preferred).')
    parser.add_argument('--rtdetr_config', type=str, required=True, help='Path to RT-DETR configuration file (.yaml).')
    parser.add_argument('--rtdetr_resume', type=str, required=True, help='Path to RT-DETR checkpoint file (.pth or .pt).')

    # --- Model & Inference Config ---
    parser.add_argument('--config', type=str, default='medium', choices=["nano", "small", "medium", "large"], help='TwinLiteNetPlus model configuration used for the weights.')
    parser.add_argument('--img_size', type=int, nargs=2, default=[352, 640], help='Image size (H W) for model input.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use (e.g., "cuda:0", "cpu").')
    parser.add_argument('--half', action='store_true', help='Use half precision (FP16) for inference (CUDA only).')

    args = parser.parse_args()

    inference(args)
    print("Inference finished.")
