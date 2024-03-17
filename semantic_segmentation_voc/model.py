from torch import nn
from torchvision.models import vgg16_bn, VGG16_BN_Weights


class CustomCNNEncoderDecoder(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
    
        # encoder
        self._encoder = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT).features
        
        self._conv_block_1 = nn.Sequential()
        self._conv_block_2 = nn.Sequential()
        self._conv_block_3 = nn.Sequential()
        self._conv_block_4 = nn.Sequential()
        self._conv_block_5 = nn.Sequential()
        
        for i in range(0, 7): # First maxpooling
            self._conv_block_1.add_module(str(i), self._encoder[i])
        for i in range(7, 14): # Second maxpooling
            self._conv_block_2.add_module(str(i), self._encoder[i])
        for i in range(14, 24): # ...
            self._conv_block_3.add_module(str(i), self._encoder[i])
        for i in range(24, 34): # ...
            self._conv_block_4.add_module(str(i), self._encoder[i])
        for i in range(34, 44): # ...
            self._conv_block_5.add_module(str(i), self._encoder[i])
        
        # decoder        
        self._deconv_block_1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self._deconv_block_2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        
        self._deconv_block_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self._deconv_block_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self._deconv_block_5 = nn.Sequential(
            nn.ConvTranspose2d(64, num_classes, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(),
            nn.Conv2d(num_classes, num_classes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
    
    def forward(self, x):
        xc1 = self._conv_block_1(x)
        xc2 = self._conv_block_2(xc1)
        xc3 = self._conv_block_3(xc2)
        xc4 = self._conv_block_4(xc3)
        xc5 = self._conv_block_5(xc4)

        xd1 = self._deconv_block_1(xc5)
        xd2 = self._deconv_block_2(xd1 + xc4)
        xd3 = self._deconv_block_3(xd2 + xc3)
        xd4 = self._deconv_block_4(xd3 + xc2)
        xd5 = self._deconv_block_5(xd4 + xc1)
    
        return xd5