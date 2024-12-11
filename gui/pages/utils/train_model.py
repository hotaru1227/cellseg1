import os

from data.dataset import TrainDataset
from gui.pages import run_prediction


def load_dataset(config):
    train_dataset = TrainDataset(
        image_dir=config["train_image_dir"],
        mask_dir=config["train_mask_dir"],
        resize_size=config["resize_size"],
        patch_size=config["patch_size"],
        train_id=config["train_id"],
        duplicate_data=config["duplicate_data"],
    )
    return train_dataset


def evaluate(model, config,state_manager):

    run_prediction(config, state_manager)
    return 1

def train_model(config, state_manager):
    os.environ["CUDA_VISIBLE_DEVICES"] = config["selected_gpu"]

    from cellseg1_train import (
        load_model,
        prepare_directories,
        save_model_pth,
        setup_training,
        train_epoch,
    )
    from set_environment import set_env

    set_env(
        config["deterministic"],
        config["seed"],
        config["allow_tf32_on_cudnn"],
        config["allow_tf32_on_matmul"],
    )
    prepare_directories(config)

    train_dataset = load_dataset(config)

    model = load_model(config)

    trainloader, optimizer, scheduler = setup_training(config, model, train_dataset)

    save_model = config["result_pth_path"]

    max_aji = 0
    max_pq = 0
    try:
        for epoch in range(config["epoch_max"]):
            if state_manager.check_stop_flag():
                save_model = False
                break

            train_epoch(model, config, trainloader, optimizer, scheduler)
            progress = int(((epoch + 1) / config["epoch_max"]) * 100)
            current_epoch = epoch + 1
            print("current_epoch:",current_epoch)
            state_manager.save_progress(progress, current_epoch)

            if current_epoch ==1 or current_epoch % 10 == 0:
                aji,pq = evaluate(model, config, state_manager)
                if aji>max_aji:
                    max_aji = aji
                    save_model_pth(model, config['checkpoint_path']+"/bestaji.pth")
                if pq>max_pq:
                    max_pq = pq
                    save_model_pth(model, config['checkpoint_path']+"/bestpq.pth")
        if save_model:
            save_model_pth(model, config["result_pth_path"])
    finally:
        state_manager.clear_training_state()
