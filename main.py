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
import math
from itertools import chain, combinations
import numpy as np

batch_size = 32
learning_rate = 5e-3
timesteps = 8
epochs = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if os.path.exists("data/FerDVS"):
    data_dir = "data"
    timesteps = 12
    ckpt = "experiments/snn_dvsgesture.pt"
    
    if not os.path.exists(ckpt):
        ckpt = -1
else:
    data_dir = "/datas/sandbox"
    ckpt = None


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_number: int,
    dataset: str,
    trans: list,
    mode="snn",
):
    module = FerModule(
        learning_rate=learning_rate,
        timesteps=timesteps,
        n_classes=6,
        epochs=epochs,
        mode=mode,
    )
    
    if ckpt is not None and ckpt != -1:
        print('Load CHECKPOINT')
        module.model.encoder.load_state_dict(torch.load(ckpt), strict=False)
    

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
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_acc", mode="max", patience=50),
        ],
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

    report = open(f"report_{dataset}.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} MODE={mode} DATASET={dataset} FOLD={fold_number} ACC={checkpoint_callback.best_model_score} TRANS={trans}\n"
    )
    report.flush()
    report.close()
    return checkpoint_callback.best_model_score


def compare(mode: str = "snn", trans: list = []):
    transform = DVSTransform(
        sensor_size=FerDVS.sensor_size,
        timesteps=timesteps,
        transforms_list=trans,
        concat_time_channels="cnn" in mode,
    )

    glob_accs = []

    for dataset in FerDVS.available_datasets:
        accs = np.zeros(10, dtype=float)
        for i in range(10):
            fold_number = i
            train_set = FerDVS(
                save_to=data_dir,
                dataset=dataset,
                train=True,
                fold=fold_number,
                transform=transform,
            )
            train_workers = math.ceil(len(train_set) / batch_size)
            train_loader = DataLoader(
                train_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=train_workers,
            )

            val_set = FerDVS(
                save_to=data_dir,
                dataset=dataset,
                train=False,
                fold=fold_number,
                transform=DVSTransform(
                    FerDVS.sensor_size,
                    timesteps=timesteps,
                    transforms_list=[],
                    concat_time_channels="cnn" in mode,
                ),
            )
            val_workers = math.ceil(len(val_set) / batch_size)
            val_loader = DataLoader(
                val_set, batch_size=batch_size, shuffle=False, num_workers=val_workers
            )

            print(f"\n\nEXPERIENCE FOR DATASET={dataset} FOLD={fold_number}")
            print(f"|TRAIN SET|={len(train_set)}")
            print(f"|VAL SET|={len(val_set)}")

            acc = train(
                train_loader, val_loader, fold_number, dataset, trans, mode=mode
            )
            accs[fold_number] = acc
            glob_accs.append(acc)

        report = open(f"report_{dataset}.txt", "a")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        report.write(f"{dt_string} Mean of folds = {accs.mean()}\n\n")
        report.flush()
        report.close()

    return sum(glob_accs) / len(glob_accs)


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


if __name__ == "__main__":
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)

    mode = "snn"
    if ckpt == -1:
        mode = "cnn"
    else:
        mode = "snn"
    
    # poss_trans = list(
    #     powerset(["flip", "background_activity", "reverse", "flip_polarity", "crop"])
    # )
    # print(len(poss_trans))

    # best_acc = -1
    # best_tran = []
    # for curr in poss_trans:
    #     acc = compare(mode=mode, trans=list(curr))
    #     if acc >= best_acc:
    #         best_acc = acc
    #         best_tran = list(curr)

    # print("BEST TRANS IS", best_tran)
    
    

    # curr = ['flip', 'background_activity', 'flip_polarity', 'transrot']
    # compare(mode=mode, trans=curr)

    curr = ['flip', 'background_activity', 'flip_polarity', 'event_drop_2']
    compare(mode=mode, trans=curr)

    # curr = ['flip', 'background_activity', 'flip_polarity', 'transrot', 'event_drop_2']
    # compare(mode=mode, trans=curr)
