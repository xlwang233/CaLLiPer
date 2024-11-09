import os
import argparse
import yaml
from easydict import EasyDict as edict

import numpy as np
import random
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, Timer
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from model import LocCLIPLightning
from dataset import POIDataModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/grid_llama3.yaml")    
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    config = edict(config)

    pe_name = config["model"]["location_encoder"]["pe_type"]
    nn_name = config["model"]["location_encoder"]["nn_type"]
    txt_enc_name = config["model"]["text_encoder"]
    if pe_name == "sphericalharmonics":
        legendre_polys = config["model"]["location_encoder"]["legendre_polys"]
    else:
        min_lambda = config["model"]["location_encoder"]["min_lambda"]
        max_lambda = config["model"]["location_encoder"]["max_lambda"]
        frequency_num = config["model"]["location_encoder"]["frequency_num"]
    hidden_dim = config["model"]["location_encoder"]["dim_hidden"]
    embed_dim = config["model"]["location_encoder"]["dim_output"]

    if pe_name == "sphericalharmonics":
        expname = f"{pe_name}-{nn_name}-{txt_enc_name}-l{legendre_polys}-h{hidden_dim}-e{embed_dim}-bs{config['training']['batch_size']}-lr{config['training']['learning_rate']}"
    else:
        expname = f"{pe_name}-{nn_name}-{txt_enc_name}-r{min_lambda}-R{max_lambda}-f{frequency_num}-h{hidden_dim}-e{embed_dim}-bs{config['training']['batch_size']}-lr{config['training']['learning_rate']}"

    lightning_config = config["lightning"]

    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
    random.seed(config["training"]["seed"])

    model = LocCLIPLightning(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params}")
    data_module = POIDataModule(config)

    timer = Timer()

    callbacks = [
        # EarlyStopping(monitor="val_loss", mode="min", patience=hparams["patience"]),
        timer
    ]

    if lightning_config.save_model:
        callbacks += [ModelCheckpoint(
            dirpath=os.path.join(lightning_config.logdir, expname),
            monitor='val_loss',
            filename="{epoch}-{val_loss:.3f}",
            save_last=True,
            save_top_k=1  # save top k best models
        )]
    
    if lightning_config.logger == "tensorboard":
        logger = TensorBoardLogger(lightning_config.logdir, 
                                   name=expname)
    elif lightning_config.logger == "wandb":
        logger = WandbLogger(project="locclip",
                             name=expname)

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        log_every_n_steps=100,
        callbacks=callbacks,
        accelerator=lightning_config.accelerator,
        devices=lightning_config.devices,
        logger=logger
    )
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    # config_fn = "configs/default.yaml"
    main()
