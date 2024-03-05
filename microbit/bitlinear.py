import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Learnable parameters for quantization and dequantization
        self.gamma = nn.Parameter(torch.ones(in_features))
        self.beta = nn.Parameter(torch.ones(out_features))

    def forward(self, input):
        # Absmax Quantization
        quant_scale = torch.max(torch.abs(input), dim=1, keepdim=True).values
        input_quant = torch.sign(input) * (quant_scale / self.gamma)

        # 1-bit Weights Quantization
        weight_quant = torch.sign(self.weight)

        # MatMul with 1-bit weights
        output = torch.matmul(input_quant, weight_quant.t())

        # Adding bias using broadcasting
        if self.bias is not None:
            output += self.bias

        # Dequantization with learnable parameters
        output *= self.beta.unsqueeze(0)

        return output
