import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, bias=True):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
    )

def deconv3x3(in_channels, out_channels, stride=2):
    return nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1
    )

def relu(inplace=True):
    return nn.ReLU(inplace=inplace)

def bn3d(num_features):
    return nn.BatchNorm3d(num_features=num_features)

def norm(num_features):
    return nn.InstanceNorm3d(num_features=num_features)

def maxpool2x2(stride):
    return nn.MaxPool3d(kernel_size=2, stride=stride, padding=0)

class ConvBlock(nn.Module):
    def __init__(self, num_layer, in_channels, out_channels, max_pool=False):
        super(ConvBlock, self).__init__()

        layers = []
        for i in range(num_layer):
            _in_channels = in_channels if i == 0 else out_channels
            layers.append(conv3x3(_in_channels, out_channels))
            layers.append(norm(out_channels))
            layers.append(relu())

        if max_pool:
            layers.append(maxpool2x2(stride=2))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    
class OccPredictor3d(nn.Module):
    def __init__(self, n_input, n_output):
        super(OccPredictor3d, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        # encoder blocks
        encoder_num_layers = [2, 2, 2, 2]
        encoder_num_filters = [8, 16, 64, 128]
        _in_channels = self.n_input
        self.block1 = ConvBlock(
            encoder_num_layers[0], _in_channels, encoder_num_filters[0], max_pool=True
        )
        self.block2 = ConvBlock(
            encoder_num_layers[1], encoder_num_filters[0], encoder_num_filters[1], max_pool=True
        )
        self.block3 = ConvBlock(
            encoder_num_layers[2], encoder_num_filters[1], encoder_num_filters[2], max_pool=True
        )
        self.block4 = ConvBlock(encoder_num_layers[3], encoder_num_filters[2], encoder_num_filters[3], max_pool=True)

        # decoder blocks
        # self.decode1 = deconv3x3(encoder_num_filters[3], encoder_num_filters[2])
        # self.decode2 = deconv3x3(encoder_num_filters[2], encoder_num_filters[1])
        # self.decode3 = deconv3x3(encoder_num_filters[1], encoder_num_filters[0])
        # self.decode4 = deconv3x3(encoder_num_filters[0], n_output)
        # axu: add relu
        # self.decode1 = relu(deconv3x3(encoder_num_filters[3], encoder_num_filters[2]))
        # self.decode2 = relu(deconv3x3(encoder_num_filters[2], encoder_num_filters[1]))
        # self.decode3 = relu(deconv3x3(encoder_num_filters[1], encoder_num_filters[0]))
        # self.decode4 = relu(deconv3x3(encoder_num_filters[0], n_output))
        self.decode1 = nn.Sequential(deconv3x3(encoder_num_filters[3], encoder_num_filters[2]), norm(encoder_num_filters[2]), relu())
        self.decode2 = nn.Sequential(deconv3x3(encoder_num_filters[2], encoder_num_filters[1]), norm(encoder_num_filters[1]), relu())
        self.decode3 = nn.Sequential(deconv3x3(encoder_num_filters[1], encoder_num_filters[0]), norm(encoder_num_filters[0]), relu())
        self.decode4 = nn.Sequential(deconv3x3(encoder_num_filters[0], n_output), relu())

        # output linear
        self.linear = conv3x3(n_output, n_output)

    def forward(self, x):
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        
        _, _, _h, _w, _l  = f3.size()
        d3 = f3 + F.interpolate(self.decode1(f4), size=(_h, _w, _l), mode="trilinear", align_corners=True)
        
        _, _, _h, _w, _l  = f2.size()
        d2 = f2 + F.interpolate(self.decode2(d3), size=(_h, _w, _l), mode="trilinear", align_corners=True)

        _, _, _h, _w, _l  = f1.size()
        d1 = f1 + F.interpolate(self.decode3(d2), size=(_h, _w, _l), mode="trilinear", align_corners=True)

        _, _, _h, _w, _l  = x.size()
        out = self.linear(x) + F.interpolate(self.decode4(d1), size=(_h, _w, _l), mode="trilinear", align_corners=True)

        F.relu(out, inplace=True)

        return out