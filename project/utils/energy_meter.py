import os
from spikingjelly.clock_driven import functional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data.dataloader import DataLoader
from project.models.sew_resnet import MultiStepSEWResNet

# see Table IV of https://arxiv.org/pdf/2110.07742.pdf
ENERGY_CMOS_45NM = {
    'E_MULT': 3.7,
    'E_ADD': 0.9,
    'E_MAC': 4.6,  # FP_MULT + E_ADD
    'E_AC': 0.9
}


class EnergyMeter(object):
    """Computes the energy consumption of a spike in parameters following the method in https://arxiv.org/pdf/2110.07742.pdf"""

    def __init__(self, layer: nn.Module, C_in: int, C_out: int, k: int = 1, O: int = 1):
        """

        Args:
            layer (nn.Module): The layer that is used to compute the spikes
            C_in (int): the number of input channels
            C_out (int): The number of output channels
            k (int, optional): The kernel size (for conv layers). Set to 1 for FC layer. Defaults to 1.
            O (int, optional): Output feature map size (HxW). Set to 1 for FC Layer. Defaults to 1.
        """
        super(EnergyMeter, self).__init__()

        self.hook = layer.register_forward_hook(self.hook_save_spikes)
        self.neuron_number = None  # initialized at the first hook call
        self.spike_count = 0
        self.C_in = float(C_in)
        self.C_out = float(C_out)
        self.k = float(k)
        self.O = float(O)

        # accumulators
        self.total_flops_snn = 0.
        self.total_flops_ann = 0.
        self.total_E_ANN = 0.
        self.total_E_SNN = 0.
        self.total_spike_rate = 0.

    def hook_save_spikes(self, module, input, output):
        spikes = output.detach().cpu().numpy()
        self.spike_count = np.count_nonzero(spikes)
        self.neuron_number = spikes.size
        self.get_energy()

    # def hook_save_spikes(self, module, input, output):

    #     spikes = output[0].detach().cpu().numpy()
    #     if self.neuron_number is None:
    #         self.neuron_number = spikes.size
    #         self.spike_count = np.count_nonzero(spikes)
    #     else:
    #         self.spike_count = self.spike_count + np.count_nonzero(spikes)

    def get_energy(self):
        spike_rate = self.spike_count / self.neuron_number

        # GET FLOPS
        flops_ann = (self.k**2) * (self.O**2) * self.C_in * self.C_out
        flops_snn = flops_ann * spike_rate

        # GET Energy consumption
        E_ANN = flops_ann * ENERGY_CMOS_45NM['E_MAC']
        E_SNN = flops_snn * ENERGY_CMOS_45NM['E_AC']

        # reinitialize after computation
        self.neuron_number = 0
        self.spike_count = 0

        # add to total
        self.total_flops_snn += flops_snn
        self.total_flops_ann += flops_ann
        self.total_E_ANN += E_ANN
        self.total_E_SNN += E_SNN
        self.total_spike_rate += spike_rate

        return flops_snn, flops_ann, E_ANN, E_SNN, spike_rate