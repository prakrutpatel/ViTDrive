import torch
import numpy as np
from IOUEval import SegmentationMetric
import logging
import logging.config
from tqdm import tqdm
import os
import torch.nn as nn
from const import *
import yaml
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F # Added for potential resizing needed in val_one
import torchvision.transforms as T # Add this import if not already present
import wandb # Make sure wandb is imported or accessible

# --- Distributed Imports (needed for is_initialized check) ---
import torch.distributed as dist
# --- End Distributed Imports ---


LOGGING_NAME="custom"
def set_logging(name=LOGGING_NAME, verbose=True):
    # sets up logging for the given name
    rank = int(os.getenv('RANK', -1))  # rank in world for Multi-GPU trainings
    level = logging.INFO if verbose and rank in {-1, 0} else logging.ERROR
    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            name: {
                'format': '%(message)s'}},
        'handlers': {
            name: {
                'class': 'logging.StreamHandler',
                'formatter': name,
                'level': level,}},
        'loggers': {
            name: {
                'level': level,
                'handlers': [name],
                'propagate': False,}}})
set_logging(LOGGING_NAME)  # run before defining LOGGER
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def poly_lr_scheduler(args, hyp, optimizer, epoch, power=1.5):
    lr = round(hyp['lr'] * (1 - epoch / args.max_epochs) ** power, 8)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(args, train_loader, model, rtdetr_model, criterion, optimizer, epoch, scaler, device,
          use_wandb, ema=None, model_for_ema_update=None, rank=0, world_size=1):
    """
    Train the model for one epoch.
    Args:
        model: The model to train (potentially DDP wrapped).
        ema: The EMA instance (if used).
        model_for_ema_update: The underlying model (unwrapped) for EMA updates.
        rank: Current process rank.
        world_size: Total number of processes.
        ...
    """
    model.train() # Ensure model (potentially DDP) is in training mode
    rtdetr_model.eval() # Ensure RT-DETR is in eval mode
    is_main_process = (rank == 0)
    verbose = args.verbose and is_main_process # Only show progress bar on main process

    if is_main_process: print(f"Starting Training Epoch: {epoch}")
    total_batches = len(train_loader)
    pbar = enumerate(train_loader)
    batch_num = 0 # Counter for global step within the epoch

    if verbose:
        LOGGER.info( ('\n' + '%13s' * 4) % ('Epoch','TverskyLoss','FocalLoss' ,'TotalLoss') )
        pbar = tqdm(pbar, total=total_batches, bar_format='{l_bar}{bar:10}{r_bar}', desc=f"Epoch {epoch} Train")

    # Zero-out gradients before starting the epoch
    optimizer.zero_grad(set_to_none=True)

    for i, (_, input, target) in pbar:
        global_step = epoch * total_batches + i # Calculate global step for logging

        input = input.to(device, non_blocking=True).float() / 255.0
        B, C, H, W = input.shape

        # Move targets to device
        target_on_device = []
        if isinstance(target, (list, tuple)):
            target_on_device = [t.to(device, non_blocking=True) for t in target if isinstance(t, torch.Tensor)]
        elif isinstance(target, torch.Tensor):
             target_on_device = target.to(device, non_blocking=True)

        # --- Generate Heatmap (using the RT-DETR model on the current device) ---
        # generate_rtdetr_heatmap is already @torch.no_grad()
        heatmap = generate_rtdetr_heatmap(rtdetr_model, input, target_size=(H, W), device=device)

        # --- Forward Pass (using potentially DDP-wrapped model) ---
        # Autocast for mixed precision
        with torch.amp.autocast(device_type=device.type, enabled=args.onGPU):
            output = model(input, heatmap)
            focal_loss, tversky_loss, loss = criterion(output, target_on_device)

        # Check for NaN loss
        if torch.isnan(loss):
             print(f"Warning: NaN loss detected at Epoch {epoch}, Batch {i}, Rank {rank}. Skipping batch.")
             # Gradients should be zeroed before next batch by optimizer.zero_grad()
             # Need to skip optimizer step and EMA update for this batch
             # Reset gradients explicitly just in case
             optimizer.zero_grad(set_to_none=True)
             continue

        # --- Backward Pass & Optimizer Step ---
        scaler.scale(loss).backward()
        # Gradient accumulation can be added here if desired
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True) # Zero grads *after* optimizer step

        # --- EMA Update (only if EMA is enabled) ---
        if ema is not None:
            ema.update(model_for_ema_update) # Update EMA using the unwrapped model state

        # --- Logging (only on main process) ---
        if is_main_process:
            # Log Batch Metrics to Wandb
            if use_wandb:
                log_data = {
                    'train/total_loss_batch': loss.item(),
                    'train/tversky_loss_batch': tversky_loss if isinstance(tversky_loss, float) else tversky_loss.item(),
                    'train/focal_loss_batch': focal_loss if isinstance(focal_loss, float) else focal_loss.item(),
                    'train/lr': optimizer.param_groups[0]['lr'] # Log learning rate
                }
                wandb.log(log_data, step=global_step)

            # Update progress bar
            if verbose:
                pbar.set_description( ('%13s' * 1 + '%13.4g' * 3) %
                                         (f'{epoch}/{args.max_epochs - 1}',
                                          tversky_loss if isinstance(tversky_loss, float) else tversky_loss.item(),
                                          focal_loss if isinstance(focal_loss, float) else focal_loss.item(),
                                          loss.item()) )
        # No return needed as EMA is updated in-place



@torch.no_grad()
def val(val_loader, model, rtdetr_model, half=False, args=None, device=None, rank=0, world_size=1):
    """
    Validate the model on the validation set.
    Args:
        model: The model to validate (should be unwrapped, e.g., ema.ema or model.module).
        rank: Current process rank.
        world_size: Total number of processes.
        ...
    Returns:
        Tuple: (da_segment_result, ll_segment_result) containing average metrics.
               Currently, these metrics are calculated *only* on rank 0's data.
               For true distributed validation, aggregation across ranks is needed.
    """
    model.eval() # Set the passed model (unwrapped) to eval mode
    rtdetr_model.eval() # Ensure RT-DETR is in eval mode
    is_main_process = (rank == 0)
    verbose = args.verbose and is_main_process

    # Initialize metric objects and average meters on all processes
    # This is needed if we want to aggregate later, but for rank 0 reporting, technically only needed there.
    DA = SegmentationMetric(2)
    LL = SegmentationMetric(2)
    da_acc_seg = AverageMeter()
    da_IoU_seg = AverageMeter()
    da_mIoU_seg = AverageMeter()
    ll_acc_seg = AverageMeter()
    ll_IoU_seg = AverageMeter()
    ll_mIoU_seg = AverageMeter()

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    if verbose:
        pbar = tqdm(pbar, total=total_batches, desc="Validation")

    # --- Loop over validation data ---
    for i, (_, input, target) in pbar:
        dtype = torch.half if half else torch.float
        input = input.to(device, non_blocking=True).to(dtype) / 255.0
        B, C, H, W = input.shape

        target_on_device = []
        if isinstance(target, (list, tuple)):
             target_on_device = [t.to(device, non_blocking=True) for t in target if isinstance(t, torch.Tensor)]
        elif isinstance(target, torch.Tensor):
             target_on_device = target.to(device, non_blocking=True)

        # Generate Heatmap
        heatmap = generate_rtdetr_heatmap(rtdetr_model, input.float(), target_size=(H, W), device=device)
        heatmap = heatmap.to(dtype)

        # Run model forward pass (already in no_grad context)
        output = model(input, heatmap)

        # --- Process outputs and targets ---
        out_da = output[0]
        target_da = target_on_device[0]
        out_ll = output[1]
        target_ll = target_on_device[1]

        # Resize outputs to match targets
        target_h, target_w = target_da.shape[-2:]
        out_da_resized = F.interpolate(out_da, size=(target_h, target_w), mode='bilinear', align_corners=False)
        _, da_predict = torch.max(out_da_resized, 1)
        _, da_gt = torch.max(target_da, 1)

        target_h_ll, target_w_ll = target_ll.shape[-2:]
        out_ll_resized = F.interpolate(out_ll, size=(target_h_ll, target_w_ll), mode='bilinear', align_corners=False)
        _, ll_predict = torch.max(out_ll_resized, 1)
        _, ll_gt = torch.max(target_ll, 1)

        # --- Update Metrics (on each process independently for now) ---
        # Note: This calculates metrics based *only* on the data processed by the current rank.
        DA.reset()
        DA.addBatch(da_predict.cpu(), da_gt.cpu()) # Metrics calculated on CPU
        da_acc_seg.update(DA.pixelAccuracy(), B)
        da_IoU_seg.update(DA.IntersectionOverUnion(), B)
        da_mIoU_seg.update(DA.meanIntersectionOverUnion(), B)

        LL.reset()
        LL.addBatch(ll_predict.cpu(), ll_gt.cpu()) # Metrics calculated on CPU
        ll_acc_seg.update(LL.lineAccuracy(), B)
        ll_IoU_seg.update(LL.IntersectionOverUnion(), B)
        ll_mIoU_seg.update(LL.meanIntersectionOverUnion(), B)

        if verbose:
            # Format numbers into strings before putting them in the dictionary
            postfix_dict = {
                'DA_mIoU': f"{da_mIoU_seg.avg:.4f}",
                'LL_Acc': f"{ll_acc_seg.avg:.4f}"
            }
            pbar.set_postfix(postfix_dict)

    # --- Aggregation Step (Placeholder - Currently Rank 0 reporting) ---
    # For true distributed validation, gather results here using e.g., dist.all_reduce
    # Example (conceptual): dist.all_reduce(da_mIoU_seg.sum, op=dist.ReduceOp.SUM)
    #                     dist.all_reduce(da_mIoU_seg.count, op=dist.ReduceOp.SUM)
    #                     global_da_mIoU = da_mIoU_seg.sum / da_mIoU_seg.count ... etc.
    if world_size > 1:
        # Barrier ensures all processes finish calculating metrics before rank 0 returns
        dist.barrier()

    # Return the average metrics calculated by the current process.
    # In the main script, only rank 0 will print/log these.
    da_segment_result = (da_acc_seg.avg, da_IoU_seg.avg, da_mIoU_seg.avg)
    ll_segment_result = (ll_acc_seg.avg, ll_IoU_seg.avg, ll_mIoU_seg.avg)

    return da_segment_result, ll_segment_result


@torch.no_grad()
def val_one(val_loader = None, model = None, half = False, args=None):

    model.eval()

    RE=SegmentationMetric(2)

    acc_seg = AverageMeter()
    IoU_seg = AverageMeter()
    mIoU_seg = AverageMeter()

    total_batches = len(val_loader)
    pbar = enumerate(val_loader)
    if args.verbose:
        pbar = tqdm(pbar, total=total_batches)
    for i, (_,input, target) in pbar:
        input = input.cuda().half() / 255.0 if half else input.cuda().float() / 255.0
        
        input_var = input
        target_var = target

        # run the mdoel
        with torch.no_grad():
            output = model(input_var)




        _,predict=torch.max(output, 1)
        predict = predict[:,12:-12]
        _,gt=torch.max(target, 1)
        
        RE.reset()
        RE.addBatch(predict.cpu(), gt.cpu())

        acc = RE.lineAccuracy()
        IoU = RE.IntersectionOverUnion()
        mIoU = RE.meanIntersectionOverUnion()

        acc_seg.update(acc,input.size(0))
        IoU_seg.update(IoU,input.size(0))
        mIoU_seg.update(mIoU,input.size(0))


    segment_result = (acc_seg.avg,IoU_seg.avg,mIoU_seg.avg)
    
    return segment_result


def save_checkpoint(state, filenameCheckpoint='checkpoint.pth.tar'):
    torch.save(state, filenameCheckpoint)

def netParams(model):
    return np.sum([np.prod(parameter.size()) for parameter in model.parameters()])

# --- generate_rtdetr_heatmap function ---
@torch.no_grad() # Ensure no gradients are calculated for RT-DETR
def generate_rtdetr_heatmap(rtdetr_model, input_batch, target_size, device):
    """
    Generates a heatmap using a pre-loaded, frozen RT-DETR model.
    Args:
        rtdetr_model: The loaded and frozen RT-DETR model object.
        input_batch: The input batch of images (B, C, H, W), scaled [0, 1].
        target_size: Tuple (H, W) for the desired final output heatmap size (matching TwinLiteNetPlus input).
        device: The torch device ('cuda' or 'cpu').
    Returns:
        torch.Tensor: The generated heatmap (B, 1, H, W), normalized [0, 1].
    """
    B, C, H, W = input_batch.shape
    heatmaps_resized = torch.zeros((B, 1, target_size[0], target_size[1]), device=device)

    # --- RT-DETR Preprocessing ---
    # Define the expected input size for RT-DETR (e.g., 640x640)
    rtdetr_expected_size = (640, 640) # Adjust if your model expects a different size

    # Resize the input batch to the size expected by RT-DETR
    rtdetr_input = F.interpolate(input_batch, size=rtdetr_expected_size, mode='bilinear', align_corners=False)

    # Optional: Add normalization if the RT-DETR backbone requires it
    # normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # rtdetr_input = normalize(rtdetr_input)

    # --- RT-DETR Feature Extraction ---
    try:
        # Pass the resized (and optionally normalized) input to the model
        features = rtdetr_model.backbone(rtdetr_input)
        encoder_output_features = rtdetr_model.encoder(features)

        # Assuming encoder outputs a list/tuple, take the first feature map
        if isinstance(encoder_output_features, (list, tuple)) and len(encoder_output_features) > 0:
            feature_map = encoder_output_features[0] # Shape: [B, C_feat, H_feat, W_feat]
        elif isinstance(encoder_output_features, torch.Tensor):
             feature_map = encoder_output_features # Adjust if output structure differs
        else:
            print("Warning: Unexpected RT-DETR encoder output structure. Using zeros heatmap.")
            return heatmaps_resized # Return zeros if features not extracted correctly

        # --- Heatmap Calculation ---
        heatmap = torch.linalg.norm(feature_map, dim=1, keepdim=True) # Shape: [B, 1, H_feat, W_feat]

        # Normalize heatmap per image in the batch
        for i in range(heatmap.shape[0]):
             min_val = torch.min(heatmap[i])
             max_val = torch.max(heatmap[i])
             if max_val > min_val:
                 heatmap[i] = (heatmap[i] - min_val) / (max_val - min_val)
             else:
                 heatmap[i] = torch.zeros_like(heatmap[i]) # Handle uniform activation

        # --- Resize Heatmap ---
        # Resize the calculated heatmap back to the target_size needed by TwinLiteNetPlus
        heatmaps_resized = F.interpolate(heatmap, size=target_size, mode='bilinear', align_corners=False)

    except AttributeError as e:
        print(f"Error: RT-DETR model missing expected attribute (e.g., 'backbone' or 'encoder'): {e}")
        print("Cannot extract features. Returning zeros heatmap.")
    except Exception as e:
        print(f"An error occurred during RT-DETR heatmap generation: {e}")
        print("Returning zeros heatmap.")


    return heatmaps_resized # Shape: [B, 1, target_size[0], target_size[1]]
# --- End of generate_rtdetr_heatmap function ---