from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import traceback
from datetime import datetime
from project.datamodules.fer_dvs import FerDVS
from project.fer_module import FerModule
from project.utils.transforms import DVSTransform
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

batch_size = 32
learning_rate = 5e-3
timesteps = 8
epochs = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "/datas/"


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_number: int,
    dataset: str,
    trans: list,
):
    module = FerModule(learning_rate=learning_rate, timesteps=timesteps, n_classes=6)

    # saves the best model checkpoint based on the accuracy in the validation set
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename=str(fold_number) + "_{epoch:03d}_{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    # create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=torch.cuda.device_count(),
        callbacks=[checkpoint_callback, EarlyStopping(monitor="val_acc", mode="max", patience=75)],
        logger=pl.loggers.TensorBoardLogger(
            "experiments/", name=f"{dataset}_{fold_number}"
        ),
        default_root_dir=f"experiments/{dataset}",
        # precision=16,
    )

    try:
        trainer.fit(module, train_loader, val_loader)
    except:
        mess = traceback.format_exc()
        report = open("errors.txt", "a")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        report.write(f"{dt_string} ===> {mess}\n=========\n\n")
        report.flush()
        report.close()
        return -1

    report = open(f"report_{dataset}_{fold_number}_{trans}.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} MODE=SNN DATASET={dataset} FOLD={fold_number} ACC={checkpoint_callback.best_model_score} TRANS={trans}\n"
    )
    report.flush()
    report.close()
    return checkpoint_callback.best_model_score


def compare(mode: str = "snn", trans: list = []):
    transform = DVSTransform(
        sensor_size=FerDVS.sensor_size,
        timesteps=timesteps,
        transforms_list=trans,
        concat_time_channels=mode == "snn",
    )

    for dataset in FerDVS.available_datasets:
        for i in range(10):
            fold_number = i
            train_set = FerDVS(
                save_to="/datas/sandbox",
                dataset=dataset,
                train=True,
                fold=fold_number,
                transform=transform,
            )
            train_loader = DataLoader(
                train_set, batch_size=batch_size, shuffle=True, num_workers=4
            )

            val_set = FerDVS(
                save_to="/datas/sandbox",
                dataset=dataset,
                train=True,
                fold=fold_number,
                transform=DVSTransform(
                    FerDVS.sensor_size,
                    timesteps=timesteps,
                    transforms_list=[],
                    concat_time_channels=mode == "snn",
                ),
            )
            val_loader = DataLoader(
                val_set, batch_size=batch_size, shuffle=False, num_workers=0
            )

            train(train_loader, val_loader, fold_number, dataset, trans)


if __name__ == "__main__":
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)
    compare(mode="snn", trans=[])
