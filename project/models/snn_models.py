from project.models import sew_resnet, base_layers
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
import torch.nn as nn


class SNNModule(nn.Module):
    """Some Information about SNNModule"""

    def __init__(self, in_channels, timesteps, output_all, n_classes):
        super(SNNModule, self).__init__()
        self.encoder = get_encoder_snn(in_channels, timesteps, output_all)

        if output_all:
            self.fc = base_layers.LinearSpike(
                512, n_classes, bias=False, neuron_model="IF"
            )
        else:
            self.fc = nn.Linear(512, n_classes, bias=False)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
        # IMPORTANT: always apply reset_net before a new forward
        functional.reset_net(self.encoder)
        functional.reset_net(self.fc)

        x = self.encoder(x)
        x = self.fc(x)

        return x


def get_encoder_snn(in_channels: int, T: int, output_all: bool):
    resnet18 = sew_resnet.MultiStepSEWResNet(
        block=sew_resnet.MultiStepBasicBlock,
        layers=[2, 2, 2, 2],
        zero_init_residual=True,
        T=T,
        cnf="ADD",
        multi_step_neuron=neuron.MultiStepIFNode,
        detach_reset=True,
        surrogate_function=surrogate.ATan(),
        output_all=output_all,
    )

    if in_channels != 3:
        resnet18.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

    return resnet18


def get_classifier(in_channels: int, out_channels: int):
    return nn.Linear(in_channels, out_channels)
