import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class TinySNN(nn.Module):
    def __init__(self, input_size=200, hidden_size=32, output_size=4, beta=0.9):
        super().__init__()

        spike_grad = surrogate.fast_sigmoid()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad, output=False)

    def forward(self, x):
        """
        x: [batch, input_size]
        Returns membrane output of final layer: [batch, output_size]
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        cur1 = self.fc1(x)
        spk1, mem1 = self.lif1(cur1, mem1)

        cur2 = self.fc2(spk1)
        _, mem2 = self.lif2(cur2, mem2)

        return mem2