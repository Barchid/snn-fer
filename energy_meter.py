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
from project.models import sew_resnet
from spikingjelly.clock_driven import functional
import numpy as np
from project.utils.energy_meter import EnergyMeter
from project.models.snn_models import SNNModule
from project.models.models import CNNModule

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


def compute_energy_dvs(model: sew_resnet.MultiStepSEWResNet, dataloader: DataLoader, height, width):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    sample_len = len(dataloader.dataset)

    O1 = float((height // 2) * (width // 2))
    O2 = float((height // 4) * (width // 4))
    O3 = float((height // 8) * (width // 8))
    O4 = float((height // 16) * (width // 16))
    O5 = float((height // 32) * (width // 32))

    # add hooks for each spike function
    conv1 = EnergyMeter(model.sn1, model.conv1.in_channels, 64, 7, O1)

    layer1_0_conv1 = EnergyMeter(model.layer1[0].sn1, 64, 64, 3, O2)
    layer1_0_conv2 = EnergyMeter(model.layer1[0].sn2, 64, 64, 3, O2)
    layer1_1_conv1 = EnergyMeter(model.layer1[1].sn1, 64, 64, 3, O2)
    layer1_1_conv2 = EnergyMeter(model.layer1[1].sn2, 64, 64, 3, O2)

    layer2_0_conv1 = EnergyMeter(model.layer2[0].sn1, 64, 128, 3, O3)
    layer2_0_conv2 = EnergyMeter(model.layer2[0].sn2, 128, 128, 3, O3)
    layer2_0_downsample = EnergyMeter(
        model.layer2[0].downsample_sn, 64, 128, 3, O3)
    layer2_1_conv1 = EnergyMeter(model.layer2[1].sn1, 128, 128, 3, O3)
    layer2_1_conv2 = EnergyMeter(model.layer2[1].sn2, 128, 128, 3, O3)

    layer3_0_conv1 = EnergyMeter(model.layer3[0].sn1, 128, 256, 3, O4)
    layer3_0_conv2 = EnergyMeter(model.layer3[0].sn2, 256, 256, 3, O4)
    layer3_0_downsample = EnergyMeter(
        model.layer3[0].downsample_sn, 128, 256, 3, O4)
    layer3_1_conv1 = EnergyMeter(model.layer3[1].sn1, 256, 256, 3, O4)
    layer3_1_conv2 = EnergyMeter(model.layer3[1].sn2, 256, 256, 3, O4)

    layer4_0_conv1 = EnergyMeter(model.layer4[0].sn1, 256, 512, 3, O5)
    layer4_0_conv2 = EnergyMeter(model.layer4[0].sn2, 512, 512, 3, O5)
    layer4_0_downsample = EnergyMeter(
        model.layer4[0].downsample_sn, 256, 512, 3, O5)
    layer4_1_conv1 = EnergyMeter(model.layer4[1].sn1, 512, 512, 3, O5)
    layer4_1_conv2 = EnergyMeter(model.layer4[1].sn2, 512, 512, 3, O5)

    fc = EnergyMeter(model.final_neurons, 512, 4)

    # forward
    i = 0
    for batch in dataloader:
        x, y = batch
        
        x = x.permute(1, 0, 2, 3, 4).to(device)

        functional.reset_net(model)

        y_hat = model(x)
        print(f'Batch n°{i}\t\t{y_hat}\t\t{fc.spike_count}')
        i+=1

    # get average of energy for all dataloader using the accumulators
    E_SNN = 0.
    E_ANN = 0.

    E_SNN += conv1.total_E_SNN
    E_ANN += conv1.total_E_ANN
    conv1_srate = conv1.total_spike_rate / sample_len

    E_SNN += layer1_0_conv1.total_E_SNN
    E_ANN += layer1_0_conv1.total_E_ANN
    layer1_0_conv1_srate = layer1_0_conv1.total_spike_rate / sample_len

    E_SNN += layer1_0_conv2.total_E_SNN
    E_ANN += layer1_0_conv2.total_E_ANN
    layer1_0_conv2_srate = layer1_0_conv2.total_spike_rate / sample_len

    E_SNN += layer1_1_conv1.total_E_SNN
    E_ANN += layer1_1_conv1.total_E_ANN
    layer1_1_conv1_srate = layer1_1_conv1.total_spike_rate / sample_len

    E_SNN += layer1_1_conv2.total_E_SNN
    E_ANN += layer1_1_conv2.total_E_ANN
    layer1_1_conv2_srate = layer1_1_conv2.total_spike_rate / sample_len

    E_SNN += layer2_0_conv1.total_E_SNN
    E_ANN += layer2_0_conv1.total_E_ANN
    layer2_0_conv1_srate = layer2_0_conv1.total_spike_rate / sample_len

    E_SNN += layer2_0_conv2.total_E_SNN
    E_ANN += layer2_0_conv2.total_E_ANN
    layer2_0_conv2_srate = layer2_0_conv2.total_spike_rate / sample_len

    E_SNN += layer2_0_downsample.total_E_SNN
    E_ANN += layer2_0_downsample.total_E_ANN
    layer2_0_downsample_srate = layer2_0_downsample.total_spike_rate / sample_len

    E_SNN += layer2_1_conv1.total_E_SNN
    E_ANN += layer2_1_conv1.total_E_ANN
    layer2_1_conv1_srate = layer2_1_conv1.total_spike_rate / sample_len

    E_SNN += layer2_1_conv2.total_E_SNN
    E_ANN += layer2_1_conv2.total_E_ANN
    layer2_1_conv2_srate = layer2_1_conv2.total_spike_rate / sample_len

    E_SNN += layer3_0_conv1.total_E_SNN
    E_ANN += layer3_0_conv1.total_E_ANN
    layer3_0_conv1_srate = layer3_0_conv1.total_spike_rate / sample_len

    E_SNN += layer3_0_conv2.total_E_SNN
    E_ANN += layer3_0_conv2.total_E_ANN
    layer3_0_conv2_srate = layer3_0_conv2.total_spike_rate / sample_len

    E_SNN += layer3_0_downsample.total_E_SNN
    E_ANN += layer3_0_downsample.total_E_ANN
    layer3_0_downsample_srate = layer3_0_downsample.total_spike_rate / sample_len

    E_SNN += layer3_1_conv1.total_E_SNN
    E_ANN += layer3_1_conv1.total_E_ANN
    layer3_1_conv1_srate = layer3_1_conv1.total_spike_rate / sample_len

    E_SNN += layer3_1_conv2.total_E_SNN
    E_ANN += layer3_1_conv2.total_E_ANN
    layer3_1_conv2_srate = layer3_1_conv2.total_spike_rate / sample_len

    E_SNN += layer4_0_conv1.total_E_SNN
    E_ANN += layer4_0_conv1.total_E_ANN
    layer4_0_conv1_srate = layer4_0_conv1.total_spike_rate / sample_len

    E_SNN += layer4_0_conv2.total_E_SNN
    E_ANN += layer4_0_conv2.total_E_ANN
    layer4_0_conv2_srate = layer4_0_conv2.total_spike_rate / sample_len

    E_SNN += layer4_0_downsample.total_E_SNN
    E_ANN += layer4_0_downsample.total_E_ANN
    layer4_0_downsample_srate = layer4_0_downsample.total_spike_rate / sample_len

    E_SNN += layer4_1_conv1.total_E_SNN
    E_ANN += layer4_1_conv1.total_E_ANN
    layer4_1_conv1_srate = layer4_1_conv1.total_spike_rate / sample_len

    E_SNN += layer4_1_conv2.total_E_SNN
    E_ANN += layer4_1_conv2.total_E_ANN
    layer4_1_conv2_srate = layer4_1_conv2.total_spike_rate / sample_len

    E_SNN += fc.total_E_SNN
    E_ANN += fc.total_E_ANN
    fc_srate = fc.total_spike_rate / sample_len

    E_SNN = E_SNN / sample_len
    E_ANN = E_ANN / sample_len

    file_energy_recap = 'energy_recap.txt'
    with open(file_energy_recap, 'w') as recap:
        recap.write(f"E_SNN {E_SNN}\n")
        recap.write(f"E_ANN {E_ANN}\n\n")
        
        recap.write(f"conv1_srate {conv1_srate}\n")

        recap.write(f"layer1_0_conv1_srate {layer1_0_conv1_srate}\n")
        recap.write(f"layer1_0_conv2_srate {layer1_0_conv2_srate}\n")
        recap.write(f"layer1_1_conv1_srate {layer1_1_conv1_srate}\n")
        recap.write(f"layer1_1_conv2_srate {layer1_1_conv2_srate}\n\n")

        recap.write(f"layer2_0_conv1_srate {layer2_0_conv1_srate}\n")
        recap.write(f"layer2_0_conv2_srate {layer2_0_conv2_srate}\n")
        recap.write(f"layer2_0_downsample_srate {layer2_0_downsample_srate}\n")
        recap.write(f"layer2_1_conv1_srate {layer2_1_conv1_srate}\n")
        recap.write(f"layer2_1_conv2_srate {layer2_1_conv2_srate}\n")

        recap.write(f"layer3_0_conv1_srate {layer3_0_conv1_srate}\n")
        recap.write(f"layer3_0_conv2_srate {layer3_0_conv2_srate}\n")
        recap.write(f"layer3_0_downsample_srate {layer3_0_downsample_srate}\n")
        recap.write(f"layer3_1_conv1_srate {layer3_1_conv1_srate}\n")
        recap.write(f"layer3_1_conv2_srate {layer3_1_conv2_srate}\n")

        recap.write(f"layer4_0_conv1_srate {layer4_0_conv1_srate}\n")
        recap.write(f"layer4_0_conv2_srate {layer4_0_conv2_srate}\n")
        recap.write(f"layer4_0_downsample_srate {layer4_0_downsample_srate}\n")
        recap.write(f"layer4_1_conv1_srate {layer4_1_conv1_srate}\n")
        recap.write(f"layer4_1_conv2_srate {layer4_1_conv2_srate}\n")

        recap.write(f"fc_srate {fc_srate}\n")


if __name__ == "__main__":
    # seeds the random from numpy, pytorch, etc for reproductibility
    pl.seed_everything(1234)
    
    height = width = 128

    mode = "snn"
    curr = []
    
    module = FerModule.load_from_checkpoint(
        "experiments/CKPlusDVS_3/version_42/checkpoints/3_epoch=199_val_acc=0.9577.ckpt"
    )
    
    # model = SNNModule(2, timesteps=timesteps, n_classes=6, output_all=False)
    model = module.model
    dataset = FerDVS(
        save_to=data_dir,
        dataset="CKPlusDVS",
        train=False,
        fold=3,
        transform=DVSTransform(
            FerDVS.sensor_size,
            timesteps,
            [],
            concat_time_channels=False
        )
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    
    compute_energy_dvs(model.encoder, loader, 128, 128)