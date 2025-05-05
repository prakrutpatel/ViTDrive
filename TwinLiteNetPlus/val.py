import torch
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import yaml
import os
from argparse import ArgumentParser
from pathlib import Path
import torch.nn.functional as F
import time
from tqdm import tqdm

from RT_DETR.rtdetrv2_pytorch.src.core import YAMLConfig
from model.model import TwinLiteNetPlus
from utils import val, netParams
from loss import TotalLoss
import BDD100K
from IOUEval import SegmentationMetric
from utils import AverageMeter, generate_rtdetr_heatmap


def validation(args):
    """
    Perform model validation on the BDD100K dataset using a trained checkpoint.
    :param args: Parsed command-line arguments.
    """
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    args.onGPU = cuda_available

    # --- Load RT-DETR Model ---
    print("=> Loading RT-DETR model for heatmap generation...")
    try:
        rtdetr_cfg = YAMLConfig(args.rtdetr_config)
        rtdetr_checkpoint = torch.load(args.rtdetr_resume, map_location='cpu')
        if 'ema' in rtdetr_checkpoint:
            rtdetr_state = rtdetr_checkpoint['ema']['module']
        elif 'model' in rtdetr_checkpoint:
             rtdetr_state = rtdetr_checkpoint['model']
        else:
            rtdetr_state = rtdetr_checkpoint
            print("Warning: Could not find 'ema' or 'model' key in RT-DETR checkpoint. Loading the whole checkpoint.")

        rtdetr_model = rtdetr_cfg.model
        rtdetr_model.load_state_dict(rtdetr_state)
        print("=> RT-DETR model loaded successfully.")

        rtdetr_model = rtdetr_model.to(device)
        rtdetr_model.eval()

        print("=> Freezing RT-DETR parameters...")
        for param in rtdetr_model.parameters():
            param.requires_grad = False
        print("=> RT-DETR parameters frozen.")

    except FileNotFoundError:
        print(f"ERROR: RT-DETR config file not found at '{args.rtdetr_config}'")
        return
    except KeyError as e:
        print(f"ERROR: Key error loading RT-DETR checkpoint '{args.rtdetr_resume}': {e}")
        return
    except Exception as e:
        print(f"ERROR: Failed to load RT-DETR model: {e}")
        return

    # --- Load TwinLiteNetPlus Model ---
    print("=> Initializing TwinLiteNetPlus model...")
    model = TwinLiteNetPlus(args)
    model = model.to(device)
    if cuda_available:
        cudnn.benchmark = True

    # --- Load Checkpoint ---
    print(f"=> Loading checkpoint: {args.checkpoint}")
    if not os.path.isfile(args.checkpoint):
        print(f"ERROR: Checkpoint file not found at '{args.checkpoint}'")
        return

    try:
        # Load the full checkpoint dictionary
        checkpoint = torch.load(args.checkpoint, map_location=device)

        # --- Load Model State (Prefer EMA if available) ---
        if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
            print("   Loading EMA model state from checkpoint.")
            # Remove 'module.' prefix if it exists (from DDP saving)
            ema_state = {k.replace('module.', ''): v for k, v in checkpoint['ema_state_dict'].items()}
            model.load_state_dict(ema_state, strict=True) # Use EMA state
        elif 'state_dict' in checkpoint:
            print("   Loading standard model state from checkpoint.")
            # Remove 'module.' prefix if it exists (from DDP saving)
            base_state = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(base_state, strict=True) # Use base model state
        else:
            print("ERROR: Checkpoint does not contain 'ema_state_dict' or 'state_dict'. Cannot load model weights.")
            return

        print("=> Model weights loaded successfully from checkpoint.")
        start_epoch = checkpoint.get('epoch', 'N/A') # Get epoch info if available
        print(f"   Checkpoint epoch: {start_epoch}")

    except Exception as e:
        print(f"ERROR: Failed to load checkpoint or model state: {e}")
        return

    # Load hyperparameters from YAML file (needed for dataset)
    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)

    # Create validation data loader
    val_dataset = BDD100K.Dataset(hyp, valid=True)
    valLoader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Print model parameter count
    print(f'Total loaded TwinLiteNetPlus network parameters: {netParams(model)}')

    # --- Perform Validation ---
    model.eval()
    print("=> Starting validation...")

    # --- Timing Initialization ---
    num_warmup_batches = 5 # Number of batches to run before starting timer
    total_inference_time = 0
    total_images = 0
    # --- End Timing Initialization ---

    # Call utils.val, passing both models and device
    # *** NOTE: The current utils.val function needs modification ***
    # *** to return timing information or perform timing internally. ***
    # *** For now, let's add timing logic directly in *this* script ***
    # *** assuming we are doing validation batch by batch here. ***

    # --- Replicate validation loop here for timing control ---
    DA = SegmentationMetric(2)
    LL = SegmentationMetric(2)
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()

    pbar = tqdm(valLoader, desc="Validation") if args.verbose else valLoader
    for i, (_, input, target) in enumerate(pbar):
        input = input.to(device)
        B, C, H, W = input.shape # Get Batch size
        input = input.float() / 255.0 # Scale after getting shape

        target_on_device = []
        if isinstance(target, (list, tuple)):
             target_on_device = [t.to(device, non_blocking=True) for t in target if isinstance(t, torch.Tensor)]
        elif isinstance(target, torch.Tensor):
             target_on_device = target.to(device, non_blocking=True)

        # Apply half precision if enabled
        dtype = torch.half if args.half and cuda_available else torch.float
        input = input.to(dtype)

        # --- Timing Starts Here (after warmup) --- 
        if i >= num_warmup_batches:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start_time = time.time()

        # --- Generate Heatmap (Includes RT-DETR Forward Pass) ---
        # generate_rtdetr_heatmap already uses torch.no_grad internally
        heatmap = generate_rtdetr_heatmap(rtdetr_model, input.float(), target_size=(H, W), device=device) # Heatmap gen needs float
        heatmap = heatmap.to(dtype) # Convert heatmap to desired dtype

        # --- TwinLiteNetPlus Inference ---
        with torch.no_grad():
            output_da, output_ll = model(input, heatmap)

        # Record time only after warmup
        if i >= num_warmup_batches:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            total_images += B
        # --- End Timing ---

        # --- Process outputs and targets (for all batches) ---
        target_da = target_on_device[0]
        target_ll = target_on_device[1]

        # Resize outputs to match targets
        target_h, target_w = target_da.shape[-2:]
        out_da_resized = F.interpolate(output_da, size=(target_h, target_w), mode='bilinear', align_corners=False)
        _, da_predict = torch.max(out_da_resized, 1)
        _, da_gt = torch.max(target_da, 1)

        target_h_ll, target_w_ll = target_ll.shape[-2:]
        out_ll_resized = F.interpolate(output_ll, size=(target_h_ll, target_w_ll), mode='bilinear', align_corners=False)
        _, ll_predict = torch.max(out_ll_resized, 1)
        _, ll_gt = torch.max(target_ll, 1)

        # --- Update Metrics ---
        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu())
        da_acc_seg.update(DA.pixelAccuracy(), B)
        da_IoU_seg.update(DA.IntersectionOverUnion(), B)
        da_mIoU_seg.update(DA.meanIntersectionOverUnion(), B)

        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu())
        ll_acc_seg.update(LL.lineAccuracy(), B)
        ll_IoU_seg.update(LL.IntersectionOverUnion(), B)
        ll_mIoU_seg.update(LL.meanIntersectionOverUnion(), B)

        # Update progress bar if verbose
        if args.verbose:
            postfix_dict = {
                'DA_mIoU': f"{da_mIoU_seg.avg:.4f}",
                'LL_Acc': f"{ll_acc_seg.avg:.4f}"
            }
            pbar.set_postfix(postfix_dict)

    # --- Calculate and Print Final Results --- 
    # Ensure we only calculate FPS if we timed something (i.e., more batches than warmup)
    avg_fps = total_images / total_inference_time if total_inference_time > 0 and total_images > 0 else 0

    da_segment_results = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    ll_segment_results = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)

    print("--- Validation Results ---")
    print(f"  Processed {total_images} images in {total_inference_time:.4f} seconds.")
    print(f"  Average Inference Speed: {avg_fps:.2f} FPS")
    print(f"  Driving Area Segment: mIOU({da_segment_results[2]:.4f}), Acc({da_segment_results[0]:.4f}), IOU({da_segment_results[1]:.4f})")
    print(f"  Lane Line Segment:    Acc({ll_segment_results[0]:.4f}), IOU({ll_segment_results[1]:.4f}), mIOU({ll_segment_results[2]:.4f})")
    print("------------------------")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the FULL training checkpoint (.pth file)')
    parser.add_argument('--rtdetr_config', type=str, required=True, help='Path to RT-DETR configuration file (.yaml)')
    parser.add_argument('--rtdetr_resume', type=str, required=True, help='Path to RT-DETR checkpoint file (.pth or .pt)')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of parallel threads for data loading')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for validation')
    parser.add_argument('--config', type=str, default='medium', choices=["nano", "small", "medium", "large"], help='TwinLiteNetPlus model configuration')
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='Path to hyperparameters YAML file (needed for dataset)')
    parser.add_argument('--half', action='store_true', help='Use half precision (FP16) for inference')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging within the val function')

    # Parse arguments and run validation
    validation(parser.parse_args())
