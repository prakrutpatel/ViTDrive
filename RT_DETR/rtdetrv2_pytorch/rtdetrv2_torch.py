"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 
import torchvision.transforms as T

import numpy as np 
from PIL import Image, ImageDraw

from src.core import YAMLConfig


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='blue', )

        im.save(f'results_{i}.jpg')


def main(args, ):
    """main
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu') 
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # NOTE load train mode state -> convert to deploy mode
    cfg.model.load_state_dict(state)

    # Use cfg.model directly
    model = cfg.model.to(args.device)
    model.eval() # Ensure model is in evaluation mode

    im_pil = Image.open(args.im_file).convert('RGB')
    w, h = im_pil.size

    # Keep original image size for potential upsampling later
    orig_img_size = (h, w) # Note: PIL uses (w, h), torch often uses (h, w)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
        # Add normalization if required by the model backbone
        # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])
    im_data = transforms(im_pil)[None].to(args.device)

    # Get intermediate features (e.g., from backbone)
    # This assumes cfg.model has a 'backbone' attribute/module
    # The exact way to get features might differ based on model architecture
    try:
        with torch.no_grad():
            # Pass the preprocessed image data through the backbone
            features = model.backbone(im_data)
            
            # Now, pass backbone features through the HybridEncoder (assuming model.encoder)
            # The HybridEncoder takes the list of features from the backbone
            print(f"Passing {len(features)} feature maps to encoder...")
            encoder_output_features = model.encoder(features)

            # HybridEncoder outputs a list of feature maps (after FPN/PAN)
            print("Encoder output type:", type(encoder_output_features))
            if isinstance(encoder_output_features, (list, tuple)):
                print("Encoder output shapes:", [f.shape for f in encoder_output_features])
                # Select the first feature map (highest resolution from FPN/PAN)
                feature_map = encoder_output_features[0]
            elif isinstance(encoder_output_features, torch.Tensor):
                print("Encoder output shape (single tensor):", encoder_output_features.shape)
                feature_map = encoder_output_features # Assuming single tensor output
            else:
                print("Unexpected encoder output structure.")
                feature_map = None

    except AttributeError as e:
        print(f"Error: Model does not have expected attribute (e.g., 'backbone' or 'encoder'): {e}")
        print("Cannot extract intermediate features with this method.")
        print("You might need to inspect the model definition in:", args.config)
        feature_map = None
    except Exception as e:
        print(f"An error occurred during feature extraction: {e}")
        feature_map = None

    # --- Process and Visualize Heatmap ---
    if feature_map is not None:
        print(f"Selected feature map shape: {feature_map.shape}") # Shape: [B, C, H, W]

        # Assuming Batch size B=1
        heatmap = feature_map[0] # Shape: [C, H, W]

        # Reduce channel dimension C to get a single heatmap [H, W]
        # heatmap = torch.max(heatmap, dim=0)[0] # Old: Max activation across channels
        heatmap = torch.linalg.norm(heatmap, dim=0) # New: L2 norm across channels

        # Normalize heatmap to [0, 1] for better visualization
        min_val = torch.min(heatmap)
        max_val = torch.max(heatmap)
        if max_val > min_val:
            heatmap = (heatmap - min_val) / (max_val - min_val)
        else:
            heatmap = torch.zeros_like(heatmap) # Handle case of uniform activation

        # Convert to numpy array
        heatmap_np = heatmap.cpu().numpy()

        # Optional: Upsample heatmap to original image size for visualization
        # Using PIL for resizing
        heatmap_img = Image.fromarray((heatmap_np * 255).astype(np.uint8)) # Scale to 0-255 for image conversion
        # heatmap_resized = heatmap_img.resize(orig_img_size[::-1], Image.Resampling.NEAREST) # Old: Use NEAREST interpolation
        heatmap_resized = heatmap_img.resize(orig_img_size[::-1], Image.Resampling.BILINEAR) # New: Revert to BILINEAR interpolation
        heatmap_resized_np = np.array(heatmap_resized) / 255.0 # Rescale back to 0-1 if needed later, or keep as 0-255 for imshow

        # Visualize using matplotlib
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(im_pil)
            plt.title('Original Image')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(heatmap_resized_np, cmap='viridis') # 'viridis', 'hot', 'jet' are common colormaps
            plt.title('Feature Map Heatmap (L2 Norm, Encoder Output 0)') # Updated title
            plt.axis('off')

            plt.tight_layout()
            plt.savefig('heatmap_visualization.png')
            print("Saved heatmap visualization to heatmap_visualization.png")
            # plt.show() # Uncomment to display plot directly if running interactively

        except ImportError:
            print("Matplotlib not found. Cannot visualize heatmap.")
            # Fallback: Save raw heatmap data or resized heatmap image
            # np.save('heatmap_raw.npy', heatmap_np)
            # Save the resized image directly (as uint8)
            heatmap_resized.save('heatmap_resized.png')
            print("Saved resized heatmap to heatmap_resized.png")

    else:
        print("Could not generate heatmap.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, )
    parser.add_argument('-r', '--resume', type=str, )
    parser.add_argument('-f', '--im-file', type=str, )
    parser.add_argument('-d', '--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
