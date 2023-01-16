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
from torchmetrics.classification.confusion_matrix import ConfusionMatrix

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


def validate(
    val_dataset: FerDVS,
    fold_number: int,
    dataset: str,
    ckpt,
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

    if ckpt is not None:
        print("Load CHECKPOINT")
        module.load_from_checkpoint(ckpt, strict=False)

    with torch.no_grad():

        module.eval()
        try:
            for x, y in val_dataset:
                pred = module(x)

        except:
            mess = traceback.format_exc()
            report = open("errors.txt", "a")
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            report.write(f"{dt_string} ===> {mess}\n=========\n\n")
            report.flush()
            report.close()
            return -1

    report = open(f"report_{mode}_{dataset}.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} MODE={mode} DATASET={dataset} FOLD={fold_number} ACC={checkpoint_callback.best_model_score} TRANS={trans}\n"
    )
    report.flush()
    report.close()
    return checkpoint_callback.best_model_score


def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


if __name__ == "__main__":
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)

    ckpts = {"CKPlusDVS": "", "ADFESDVS": "", "CASIADVS": "", "MMIDVS": ""}

    mode = "snn"

    for dataset in FerDVS.available_datasets:
        cm = ConfusionMatrix(len(FerDVS.classes), compute_on_step=False)
        for i in range(10):
            print(dataset, f"{i}")
            fold_number = i

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
            
            val_loader = iter(val_loader)
            
            module = FerModule(
                learning_rate=learning_rate,
                timesteps=timesteps,
                n_classes=6,
                epochs=epochs,
                mode=mode,
            )

            module.load_from_checkpoint(ckpts[dataset], strict=False)
            
            with torch.no_grad():
                module.eval()
            
                for batch in val_loader:
                    x, y = batch
                    pred = module(x)
                    cm.update(pred, y)
            
            print('Fold done')
                    
        result = cm.compute()
        torch.save(result, f"experiments/confmats/{dataset}")
        