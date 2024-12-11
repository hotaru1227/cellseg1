import multiprocessing
import argparse
from datetime import datetime, timedelta
from gui.pages.utils.train_model import train_model
from gui.pages.utils.train_state_manager import TrainingStateManager
from gui.pages.utils.web_utils import (
    LORA_PTH_DIR,
    SUPPORT_EXTENSION,
    TRAIN_IMAGE_DIR,
    TRAIN_MASK_DIR,
    delete_file,
    get_available_gpus,
    get_sam_model_path,
    initialize_session_state,
    list_files,
    load_default_config,
)
from project_root import STORAGE_DIR

def prepare_config(
    epoch_max,
    base_lr,
    batch_size,
    gradient_accumulation_step,
    image_dir,
    mask_dir,
    sam_type,
    random_seed,
    sam_image_size,
    patch_size,
    resize_size,
    lora_rank,
):
    config = load_default_config()
    config["epoch_max"] = epoch_max
    config["base_lr"] = base_lr
    config["batch_size"] = batch_size
    config["gradient_accumulation_step"] = gradient_accumulation_step
    config["train_image_dir"] = str(image_dir)
    config["train_mask_dir"] = str(mask_dir)
    config["resize_size"] = [resize_size, resize_size]
    config["data_dir"] = ""
    config["vit_name"] = sam_type
    config["seed"] = random_seed
    config["model_path"] = get_sam_model_path(sam_type)
    config["sam_image_size"] = sam_image_size
    config["patch_size"] = patch_size
    config["image_encoder_lora_rank"] = lora_rank
    config["mask_decoder_lora_rank"] = lora_rank
    return config

def parse_args():
    parser = argparse.ArgumentParser('Cell prompter')

    parser.add_argument("--use_wandb", action='store_true', help='use wandb for logging')
    parser.add_argument('--run_name', default=None, type=str, help='wandb run name')

    # * Run Mode
    parser.add_argument('--eval', action='store_true')

    # * Train
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    parser.add_argument("--device_num", default="3", help="device to use for training / testing")
    parser.add_argument('--save_path', default='', type=str, help='checkpoint path.')

    # * CellSeg
    
    parser.add_argument('--epoch_max', default=200, type=int, help='Number of epochs.')
    parser.add_argument('--base_lr', default=0.001, type=float, help='Base learning rate.')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training.')
    parser.add_argument('--gradient_accumulation_step', default=1, type=int, help='Number of steps to accumulate gradients before updating.')
    parser.add_argument('--train_image_dir', default='path/to/train/images', type=str, help='Directory containing training images.')
    parser.add_argument('--train_mask_dir', default='path/to/train/masks', type=str, help='Directory containing training masks.')
    parser.add_argument('--sam_type', default='vit_b', type=str, help='Type of SAM to use.')
    parser.add_argument('--random_seed', default=42, type=int, help='Random seed for reproducibility.')
    parser.add_argument('--sam_image_size', default=256, type=int, help='Input image size for SAM.')
    parser.add_argument('--patch_size', default=256, type=int, help='Patch size for cropping images.')
    parser.add_argument('--resize_size', default=512, type=int, help='Size to resize images before processing.')
    parser.add_argument('--lora_rank', default=4, type=int, help='Rank of LoRA (Low-Rank Adaptation).')



    opt = parser.parse_args()

    return opt



if __name__ == '__main__':
    args = parse_args()

    images = list_files(TRAIN_IMAGE_DIR)
    masks = list_files(TRAIN_MASK_DIR)
    state_manager = TrainingStateManager(STORAGE_DIR)
    config = prepare_config(
                    args.epoch_max,
                    args.base_lr,
                    args.batch_size,
                    args.gradient_accumulation_step,
                    TRAIN_IMAGE_DIR,
                    TRAIN_MASK_DIR,
                    args.sam_type,
                    args.random_seed,
                    args.sam_image_size,
                    args.patch_size,
                    args.resize_size,
                    args.lora_rank,
                )
    config["selected_gpu"] = "5"
    config["train_id"] = list(range(len(images)))


    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    lora_path = STORAGE_DIR / "loras" / f"lora_{args.sam_type}_{args.lora_rank}_{args.sam_image_size}_{timestamp}.pth"
    config["result_pth_path"] = str(lora_path)
    config['checkpoint_path'] = args.save_path

    process = multiprocessing.Process(
        target=train_model,
        args=(config, state_manager),
    )
    process.start()

