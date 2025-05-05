import os
import torch
import torch.optim.lr_scheduler
import torch.backends.cudnn as cudnn
import yaml
import math
from copy import deepcopy
from argparse import ArgumentParser
import torch.nn.functional as F # Added for resizing heatmap
import numpy as np # Added for potential type conversion
import wandb # Import wandb

# --- Distributed Training Imports ---
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# --- End Distributed Training Imports ---

# Corrected import path assuming ViTDrive is in PYTHONPATH or you're running from it
from RT_DETR.rtdetrv2_pytorch.src.core import YAMLConfig

from model.model import TwinLiteNetPlus
from loss import TotalLoss
# Import the necessary functions from utils
from utils import train, val, netParams, save_checkpoint, poly_lr_scheduler, generate_rtdetr_heatmap
import BDD100K


class ModelEMA:
    """Exponential Moving Average (EMA) for model parameters"""
    def __init__(self, model, decay=0.9999, updates=0):
        # Create EMA model - IMPORTANT: EMA holds a copy of the model *before* DDP wrapping
        # No need for deepcopy if model is already on the correct device and state loaded
        self.ema = deepcopy(model).eval() # Ensure EMA model is in eval mode
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))  # Exponential decay ramp
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        """Update EMA model parameters. model is the underlying model (module if DDP)"""
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.state_dict() # Get state_dict from the model *being trained* (unwrapped)
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1. - d) * msd[k].detach()

def setup_distributed(backend='nccl', port='env://'):
    """Initializes the distributed training environment using torchrun env vars."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank(), int(os.environ.get('LOCAL_RANK', 0))

    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if world_size > 1:
        print(f"[Rank {rank}] Initializing distributed process group (backend: {backend})...")
        dist.init_process_group(backend=backend, init_method=port, world_size=world_size, rank=rank)
        torch.cuda.set_device(local_rank)
        print(f"[Rank {rank}] Process group initialized. World Size: {world_size}, Rank: {rank}, Local Rank: {local_rank}, Device: cuda:{local_rank}")
        # Barrier to ensure all processes initialize before proceeding
        dist.barrier()
    else:
        print("Distributed training not enabled (WORLD_SIZE <= 1).")

    return world_size, rank, local_rank


def train_net(args, hyp):
    """Train the neural network model with given arguments and hyperparameters"""

    # --- Distributed Setup ---
    world_size, rank, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    args.onGPU = (device.type == 'cuda') # Set onGPU flag based on actual device
    # Adjust batch size and workers based on world size if needed (or assume args are per-process)
    # args.batch_size = args.batch_size // world_size (if args.batch_size is global)
    # args.num_workers = args.num_workers // world_size

    # --- Wandb Initialization (only on main process) ---
    use_wandb = args.wandb_project is not None and is_main_process
    wandb_run_id = None
    if use_wandb:
        # Generate a run ID on the main process to allow resuming
        if not args.resume:
            wandb_run_id = wandb.util.generate_id()
        # Initialize wandb run
        # If resuming, wandb ID will be loaded from checkpoint later
        wandb.init(project=args.wandb_project,
                   name=args.wandb_run_name,
                   config=vars(args),
                   resume='allow',
                   id=wandb_run_id)
        wandb.config.update(hyp)
        wandb_run_id = wandb.run.id # Store the actual run ID
        print(f"[Rank {rank}] Weights & Biases logging enabled. Project: '{args.wandb_project}', Run ID: {wandb_run_id}")
    elif is_main_process:
        print("Weights & Biases logging disabled.")

    use_ema = args.ema

    # --- Load TwinLiteNetPlus Model (on current device) ---
    if is_main_process: print("=> Initializing TwinLiteNetPlus model...")
    model = TwinLiteNetPlus(args).to(device)

    # --- Load RT-DETR Model (Load on CPU first, then move to device) ---
    if is_main_process: print("=> Loading RT-DETR model...")
    try:
        rtdetr_cfg = YAMLConfig(args.rtdetr_config)
        rtdetr_checkpoint = torch.load(args.rtdetr_resume, map_location='cpu')
        if 'ema' in rtdetr_checkpoint:
            rtdetr_state = rtdetr_checkpoint['ema']['module']
        elif 'model' in rtdetr_checkpoint:
             rtdetr_state = rtdetr_checkpoint['model']
        else:
            rtdetr_state = rtdetr_checkpoint
            if is_main_process: print("Warning: Could not find 'ema' or 'model' key in RT-DETR checkpoint. Loading the whole checkpoint.")

        rtdetr_model = rtdetr_cfg.model
        rtdetr_model.load_state_dict(rtdetr_state)
        if is_main_process: print("=> RT-DETR model loaded successfully.")

        rtdetr_model = rtdetr_model.to(device) # Move RT-DETR to process-specific device
        rtdetr_model.eval()

        # --- Freeze RT-DETR Parameters ---
        if is_main_process: print("=> Freezing RT-DETR parameters...")
        for param in rtdetr_model.parameters():
            param.requires_grad = False
        if is_main_process: print("=> RT-DETR parameters frozen.")

    except FileNotFoundError:
        if is_main_process: print(f"ERROR: RT-DETR config file not found at '{args.rtdetr_config}'")
        if world_size > 1: dist.barrier()
        return
    except KeyError as e:
        if is_main_process: print(f"ERROR: Key error loading RT-DETR checkpoint '{args.rtdetr_resume}': {e}")
        if world_size > 1: dist.barrier()
        return
    except Exception as e:
        if is_main_process: print(f"ERROR: Failed to load RT-DETR model: {e}")
        if world_size > 1: dist.barrier()
        return

    # Create save directory on main process only
    if is_main_process:
        os.makedirs(args.savedir, exist_ok=True)
    if world_size > 1:
        dist.barrier() # Wait for main process

    # --- Initialize Training State Variables ---
    start_epoch = 0
    optimizer = None
    ema = None # EMA instance, created later if needed

    # --- Handle Weight Loading/Resuming ---
    map_location = 'cpu' # Load checkpoints to CPU first to avoid rank 0 OOM and ensure consistency

    if args.resume:
        if is_main_process: print(f"Attempting to resume training from checkpoint: {args.resume}")
        if os.path.isfile(args.resume) and args.resume.endswith(".pth"):
            try:
                checkpoint = torch.load(args.resume, map_location=map_location)

                # Load model state (handle potential DDP prefix)
                state_dict = checkpoint['state_dict']
                # Check if keys have 'module.' prefix and remove it
                if all(k.startswith('module.') for k in state_dict.keys()):
                    if is_main_process: print("   Removing 'module.' prefix from checkpoint state_dict.")
                    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                model.load_state_dict(state_dict)
                if is_main_process: print("   Model state loaded from checkpoint.")

                # Initialize optimizer BEFORE loading state
                trainable_params = filter(lambda p: p.requires_grad, model.parameters())
                optimizer = torch.optim.AdamW(trainable_params, lr=hyp['lr'], betas=(hyp['momentum'], 0.999), eps=hyp['eps'], weight_decay=hyp['weight_decay'])
                if 'optimizer' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    if is_main_process: print("   Optimizer state loaded from checkpoint.")
                elif is_main_process:
                    print("   Optimizer state not found in checkpoint. Initializing fresh optimizer.")

                # Load EMA state (if used and present)
                if use_ema:
                    if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
                        ema = ModelEMA(model, decay=hyp.get('ema_decay', 0.9999)) # Initialize EMA with current model state
                        ema_state = checkpoint['ema_state_dict']
                        if all(k.startswith('module.') for k in ema_state.keys()): # Check and remove prefix
                             if is_main_process: print("   Removing 'module.' prefix from checkpoint ema_state_dict.")
                             ema_state = {k.replace('module.', ''): v for k, v in ema_state.items()}
                        ema.ema.load_state_dict(ema_state)
                        ema.updates = checkpoint.get('updates', 0)
                        if is_main_process: print("   EMA state loaded from checkpoint.")
                    else:
                        ema = ModelEMA(model, decay=hyp.get('ema_decay', 0.9999)) # Init fresh EMA if not in checkpoint
                        if is_main_process: print("   EMA state not found in checkpoint. Initialized fresh EMA.")
                else:
                    ema = None

                # Load epoch
                start_epoch = checkpoint['epoch']
                if is_main_process: print(f"=> Successfully resumed training from epoch {start_epoch}")

                # Load and resume wandb ID on main process
                if use_wandb and 'wandb_id' in checkpoint and checkpoint['wandb_id']:
                    loaded_wandb_id = checkpoint['wandb_id']
                    print(f"[Rank {rank}] Resuming wandb run with ID: {loaded_wandb_id}")
                    wandb.init(project=args.wandb_project, id=loaded_wandb_id, resume='must')
                elif use_wandb:
                     print(f"[Rank {rank}] Warning: Could not find wandb_id in checkpoint for resuming.")

            except KeyError as e:
                if is_main_process: print(f"ERROR: Checkpoint '{args.resume}' missing key: {e}. Cannot resume.")
                if world_size > 1: dist.barrier()
                return
            except Exception as e:
                 if is_main_process: print(f"ERROR: Failed to load checkpoint '{args.resume}': {e}")
                 if world_size > 1: dist.barrier()
                 return

        else: # Resume file not found or invalid
            if is_main_process: print(f"ERROR: Checkpoint file (.pth) not found/invalid at '{args.resume}'. Cannot resume.")
            if world_size > 1: dist.barrier()
            return

    elif args.load_weights: # Finetuning: Load weights only
        if is_main_process: print(f"Attempting to load pretrained weights for finetuning: {args.load_weights}")
        if os.path.isfile(args.load_weights):
            try:
                pretrained_dict = torch.load(args.load_weights, map_location=map_location)
                # Handle potential nesting and DDP prefix
                if 'state_dict' in pretrained_dict: pretrained_dict = pretrained_dict['state_dict']
                elif 'model' in pretrained_dict: pretrained_dict = pretrained_dict['model']
                if all(k.startswith('module.') for k in pretrained_dict.keys()):
                     if is_main_process: print("   Removing 'module.' prefix from pretrained weights.")
                     pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

                missing_keys, unexpected_keys = model.load_state_dict(pretrained_dict, strict=False)
                if is_main_process:
                    print("=> Successfully loaded pretrained weights.")
                    if missing_keys: print(f"   Missing keys: {missing_keys}")
                    if unexpected_keys: print(f"   Unexpected keys: {unexpected_keys}")

                start_epoch = 0
                if is_main_process: print(f"   Starting finetuning from epoch {start_epoch}.")

            except Exception as e:
                if is_main_process: print(f"ERROR: Failed to load pretrained weights '{args.load_weights}': {e}")
                start_epoch = 0
        else:
             if is_main_process: print(f"Warning: Pretrained weights file '{args.load_weights}' not found. Starting from scratch.")
             start_epoch = 0

        # Initialize fresh optimizer and EMA for finetuning
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=hyp['lr'], betas=(hyp['momentum'], 0.999), eps=hyp['eps'], weight_decay=hyp['weight_decay'])
        if is_main_process: print("Initialized fresh optimizer for finetuning.")
        if use_ema:
            ema = ModelEMA(model, decay=hyp.get('ema_decay', 0.9999))
            if is_main_process: print("Initialized fresh EMA for finetuning.")
        else:
             ema = None

    else: # Start fresh
        if is_main_process: print("Starting training from scratch.")
        start_epoch = 0
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=hyp['lr'], betas=(hyp['momentum'], 0.999), eps=hyp['eps'], weight_decay=hyp['weight_decay'])
        if is_main_process: print("Initialized fresh optimizer.")
        if use_ema:
            ema = ModelEMA(model, decay=hyp.get('ema_decay', 0.9999))
            if is_main_process: print("Initialized fresh EMA.")
        else:
             ema = None

    # --- Wrap Model with DDP --- Must be done AFTER loading state dict and creating optimizer
    if world_size > 1:
        if is_main_process: print("Wrapping model with DistributedDataParallel...")
        # find_unused_parameters=True might be needed if some outputs aren't used in loss
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        if is_main_process: print("Model wrapped with DDP.")

    # Make sure model is on the correct device before optimizer step or training
    model.to(device)
    if ema: ema.ema.to(device) # Ensure EMA model is also on the correct device

    # Log model graph via wandb (only main process, after potential DDP wrapping)
    if use_wandb:
        # Watch the DDP model (or base model if not distributed)
        wandb.watch(model, log='all', log_freq=100)

    # --- DataLoaders with DistributedSampler ---
    if is_main_process: print("Creating datasets and dataloaders...")
    train_dataset = BDD100K.Dataset(hyp, valid=False)
    val_dataset = BDD100K.Dataset(hyp, valid=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None

    trainLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, # Assumed to be per-GPU batch size
        shuffle=(train_sampler is None), # Shuffle done by sampler if distributed
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True) # Drop last incomplete batch for consistency

    valLoader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2, # Often possible to use larger batch for validation
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False)
    if is_main_process: print("Dataloaders created.")

    if args.onGPU:
        cudnn.benchmark = True

    if is_main_process:
        # Access underlying model for param count if DDP wrapped
        model_to_measure = model.module if isinstance(model, DDP) else model
        print(f'Total TwinLiteNetPlus network parameters: {netParams(model_to_measure)}')

    criteria = TotalLoss(hyp).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=args.onGPU)

    # --- Training Loop ---
    if is_main_process: print(f"Starting training loop from epoch {start_epoch}... World Size: {world_size}")
    for epoch in range(start_epoch, args.max_epochs):
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch) # Essential for proper shuffling per epoch

        # Adjust LR (all processes do this)
        poly_lr_scheduler(args, hyp, optimizer, epoch)
        current_lr = optimizer.param_groups[0]['lr']
        if is_main_process: print(f"Epoch {epoch}/{args.max_epochs - 1} - Learning rate: {current_lr:.8f}")

        # --- Train Step ---
        # The train function needs the DDP model for forward pass, but the unwrapped model for EMA update
        model_for_ema_update = model.module if isinstance(model, DDP) else model
        train(args, trainLoader, model, rtdetr_model, criteria, optimizer, epoch, scaler, device,
              use_wandb, # Pass wandb flag (train fn checks rank)
              ema=ema, # Pass the EMA instance itself
              model_for_ema_update=model_for_ema_update, # Pass the unwrapped model for EMA
              rank=rank, world_size=world_size)

        # --- Validation Step ---
        # Perform validation potentially on all ranks, but aggregate/log on rank 0
        # Decide which model state to validate: EMA or the standard model
        model_to_validate = ema.ema if use_ema and ema is not None else model
        # If validating the DDP model itself (not EMA), access the underlying module
        if not (use_ema and ema is not None) and isinstance(model_to_validate, DDP):
            model_to_validate = model_to_validate.module

        # Run validation - val function needs to handle potential DDP communication if aggregating results
        da_segment_results, ll_segment_results = val(valLoader, model_to_validate, rtdetr_model, device=device, args=args, rank=rank, world_size=world_size)

        # --- Log Validation Metrics (only on main process) ---
        if is_main_process:
            print(f"[Rank {rank}] Validation Results Epoch {epoch}: DA mIOU={da_segment_results[2]:.4f}, LL Acc={ll_segment_results[0]:.4f}, LL IOU={ll_segment_results[1]:.4f}")
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'learning_rate': current_lr,
                    'val_da_mIOU': da_segment_results[2],
                    'val_da_Acc': da_segment_results[0],
                    'val_da_IOU': da_segment_results[1],
                    'val_ll_Acc': ll_segment_results[0],
                    'val_ll_IOU': ll_segment_results[1],
                    'val_ll_mIOU': ll_segment_results[2] # Assuming mIOU for LL is index 2
                }, step=epoch)

        # --- Save Checkpoint (only on main process) ---
        if is_main_process:
            # Always save the state_dict of the underlying model (module)
            base_model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            ema_state = ema.ema.state_dict() if use_ema and ema is not None else None
            ema_updates = ema.updates if use_ema and ema is not None else None

            checkpoint_data = {
                'epoch': epoch + 1,
                'state_dict': base_model_state,
                'ema_state_dict': ema_state,
                'updates': ema_updates,
                'optimizer': optimizer.state_dict(),
                'lr': current_lr,
                'args': args, # Save args for reproducibility
                'hyp': hyp # Save hyperparameters
            }
            if use_wandb and wandb_run_id:
                checkpoint_data['wandb_id'] = wandb_run_id

            # Determine which state to save as the standalone model file (prefer EMA)
            save_model_state = ema_state if use_ema and ema_state is not None else base_model_state
            model_file_name = os.path.join(args.savedir, f'model_epoch_{epoch}.pth')
            torch.save(save_model_state, model_file_name)
            # print(f"[Rank {rank}] Saved model state to {model_file_name}")

            # Save Full Checkpoint (Latest)
            latest_checkpoint_path = os.path.join(args.savedir, 'checkpoint_latest.pth')
            save_checkpoint(checkpoint_data, latest_checkpoint_path)
            print(f"[Rank {rank}] Saved latest checkpoint to {latest_checkpoint_path}")

            # Save Periodic Full Checkpoint
            if (epoch + 1) % args.save_interval == 0 or epoch == args.max_epochs - 1:
                 periodic_checkpoint_filename = f'checkpoint_epoch_{epoch+1:04d}.pth'
                 periodic_checkpoint_path = os.path.join(args.savedir, periodic_checkpoint_filename)
                 save_checkpoint(checkpoint_data, periodic_checkpoint_path)
                 print(f"[Rank {rank}] Saved periodic checkpoint to {periodic_checkpoint_path}")

        # --- Barrier --- Ensure all processes finish the epoch (including saving) before next one
        if world_size > 1:
            dist.barrier()

    # --- Final Cleanup --- (only on main process)
    if is_main_process:
        print("Training Finished.")
        if use_wandb: wandb.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Params
    parser.add_argument('--max_epochs', type=int, default=100, help='Max number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size PER GPU')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of dataloader workers per GPU process')
    parser.add_argument('--savedir', default='./training_results', help='Directory to save checkpoints and logs')
    parser.add_argument('--hyp', type=str, default='./hyperparameters/twinlitev2_hyper.yaml', help='Path to TwinLiteNetPlus hyperparameters YAML')
    parser.add_argument('--save_interval', type=int, default=20, help='Save a full checkpoint every N epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # Model Config
    parser.add_argument('--config', default='medium', help='TwinLiteNetPlus model configuration (e.g., nano, small)')
    parser.add_argument('--ema', action='store_true', help='Use Exponential Moving Average (EMA) for model weights')

    # Loading Weights
    parser.add_argument('--resume', type=str, default='', help='Resume training from a FULL checkpoint (.pth file)')
    parser.add_argument('--load_weights', type=str, default='', help='Load ONLY pretrained model weights (.pth file) for finetuning (starts epoch 0)')

    # RT-DETR Config
    parser.add_argument('--rtdetr_config', type=str, required=True, help='Path to RT-DETR configuration file (.yaml)')
    parser.add_argument('--rtdetr_resume', type=str, required=True, help='Path to RT-DETR checkpoint file (.pth or .pt)')

    # Logging & Verbosity
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging during train/val (only rank 0)')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name. If None, wandb is disabled.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Optional Wandb run name.')

    # --- Distributed Training Args (usually set by torchrun/slurm) ---
    # parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training (set by launcher)')

    args = parser.parse_args()

    # --- Seed and Config Validation ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Add more seeding if using other libraries like random
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.resume and args.load_weights:
        raise ValueError("Cannot use both --resume and --load_weights.")

    with open(args.hyp, errors='ignore') as f:
        hyp = yaml.safe_load(f)  # Load hyperparameters

    # --- Start Training ---
    train_net(args, hyp.copy())
