import copy
import math
import torch
import torch.nn as nn

class ConvNormAct(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 groups=1, 
                 bn=True, 
                 act=True, 
                 bias=False
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv_norm_act = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, 
                      out_channels=out_channels, 
                      kernel_size=kernel_size, 
                      stride=stride, 
                      groups=groups, 
                      padding=padding, 
                      bias=bias),
            nn.BatchNorm2d(out_channels) if bn else nn.Identity(),
            nn.SiLU(inplace=True) if act else nn.Identity()
        )
        
    def forward(self, x):
        return self.conv_norm_act(x)
        


class SEBlock(nn.Module):
    def __init__(self, in_channels, squeeze_channels):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, squeeze_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(squeeze_channels, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.se(x) * x



class StochasticDepth(nn.Module):    
    def __init__(self, survival_prob):
        super().__init__()
        self.survival_prob = survival_prob
        
    def forward(self, x):
        if not self.training:
            return x
        #[B][C][W][H]
        size = [x.shape[0]] + [1] * (x.ndim - 1)
        mask = torch.empty(size, dtype=x.dtype, device=x.device).bernoulli_(self.survival_prob)
        return x * mask.div_(self.survival_prob)
    
    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(survival_prob={self.survival_prob})"
        return s



class MBConv(nn.Module):
    def __init__(self,
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride, 
                 expansion_rate, 
                 reduction=4, 
                 survival_prob=0.8
    ):
        super().__init__()
        self.use_res_connection = stride == 1 and in_channels == out_channels
        expanded_channels = in_channels * expansion_rate
        
        self.expand = ConvNormAct(in_channels, expanded_channels, kernel_size=1) if expanded_channels != in_channels else nn.Identity()
        self.depthwise_conv = ConvNormAct(expanded_channels, expanded_channels, kernel_size=kernel_size, stride=stride, groups=expanded_channels)
        self.se = SEBlock(expanded_channels, in_channels//reduction)
        self.pointwise_conv = ConvNormAct(expanded_channels, out_channels, kernel_size=1, act=False)
        
        self.stochastic_depth = StochasticDepth(survival_prob=survival_prob)
        
    def forward(self, inputs):
        x = self.expand(inputs)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.pointwise_conv(x)
        
        if self.use_res_connection:
            x = self.stochastic_depth(x)
            x += inputs
          
        return x
        


class EfficientNet(nn.Module):
    def __init__(self, num_classes, model_blocks, dropout_rate, survival_prob=0.8):
        super().__init__()
        layers = nn.ModuleList([])
        
        first_out_channels = model_blocks[0].in_channels
        layers.append(
            ConvNormAct( 
                in_channels=3, 
                out_channels=first_out_channels,
                kernel_size=3,
                stride=2
            )
        )
        
        total_layers = sum(block.layers for block in model_blocks)
        current_layer_id = 0
        for block in model_blocks:
            stage = nn.ModuleList([])
            for _ in range(block.layers):
                block_copy = copy.copy(block)
                
                if stage:
                    block_copy.in_channels = block_copy.out_channels
                    block_copy.stride = 1
                    
                current_survival_prob = 1 - ((1 - survival_prob) * float(current_layer_id) / total_layers)

                stage.append(
                    block_copy.block(
                        in_channels=block_copy.in_channels,
                        out_channels=block_copy.out_channels, 
                        kernel_size=block_copy.kernel_size,
                        stride=block_copy.stride,
                        expansion_rate=block_copy.expansion_rate,
                        survival_prob=current_survival_prob
                    )
                )
                current_layer_id += 1
                
            layers.append(nn.Sequential(*stage))

        last_in_channels = model_blocks[-1].out_channels
        last_out_channels = last_in_channels * 4
        layers.append(
            ConvNormAct( 
                in_channels=last_in_channels, 
                out_channels=last_out_channels,
                kernel_size=1,
                stride=1
            )
        )
        
        self.num_classes = num_classes
        
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p = dropout_rate, inplace=True),
            nn.Linear(last_out_channels, num_classes)
        )
            
        
    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

class ModelBlockConfig():
    def __init__(self, block, in_channels, out_channels, layers, kernel_size, stride, expansion_rate, depth_scale, width_scale):
        self.block = block
        self.in_channels = self.make_channels_divisible_by_8(in_channels * width_scale)
        self.out_channels = self.make_channels_divisible_by_8(out_channels * width_scale)
        self.layers = self.rescale_layers(layers, depth_scale)
        self.kernel_size = kernel_size
        self.stride = stride
        self.expansion_rate = expansion_rate

    def make_channels_divisible_by_8(self, num_channels, divisor=8):
        return max(divisor, int(num_channels + divisor / 2) // divisor * divisor)

    def rescale_layers(self, layers, depth_scale):
        return int(math.ceil(layers * depth_scale))
    

class ModelCoeffsConfig():
    def __init__(self, phi, resolution, dropout):
        self.resolution = resolution
        self.dropout = dropout
        self.depth_scale = 1.2**phi
        self.width_scale = 1.1**phi


def efficientnet(version, num_classes):
    model_coeffs = {
        "B0": ModelCoeffsConfig(0, 224, 0.2),
        "B1": ModelCoeffsConfig(0.5, 240, 0.2),
        "B2": ModelCoeffsConfig(1, 260, 0.3),
        "B3": ModelCoeffsConfig(2, 300, 0.3),
        "B4": ModelCoeffsConfig(3, 380, 0.4),
        "B5": ModelCoeffsConfig(4, 456, 0.4),
        "B6": ModelCoeffsConfig(5, 528, 0.5),
        "B7": ModelCoeffsConfig(6, 600, 0.5),
    }

    depth_scale = model_coeffs[version].depth_scale
    width_scale = model_coeffs[version].width_scale

    model_blocks = [
        ModelBlockConfig(block=MBConv, in_channels=32, out_channels=16, layers=1, kernel_size=3, stride=1, expansion_rate=1, depth_scale=depth_scale, width_scale=width_scale),
        ModelBlockConfig(block=MBConv, in_channels=16, out_channels=24, layers=2, kernel_size=3, stride=2, expansion_rate=6, depth_scale=depth_scale, width_scale=width_scale),
        ModelBlockConfig(block=MBConv, in_channels=24, out_channels=40, layers=2, kernel_size=5, stride=2, expansion_rate=6, depth_scale=depth_scale, width_scale=width_scale),
        ModelBlockConfig(block=MBConv, in_channels=40, out_channels=80, layers=3, kernel_size=3, stride=2, expansion_rate=6, depth_scale=depth_scale, width_scale=width_scale),
        ModelBlockConfig(block=MBConv, in_channels=80, out_channels=112, layers=3, kernel_size=5, stride=1, expansion_rate=6, depth_scale=depth_scale, width_scale=width_scale),
        ModelBlockConfig(block=MBConv, in_channels=112, out_channels=192, layers=4, kernel_size=5, stride=2, expansion_rate=6, depth_scale=depth_scale, width_scale=width_scale),
        ModelBlockConfig(block=MBConv, in_channels=192, out_channels=320, layers=1, kernel_size=3, stride=1, expansion_rate=6, depth_scale=depth_scale, width_scale=width_scale)
    ]
    
    return EfficientNet(num_classes, model_blocks, model_coeffs[version].dropout)
      