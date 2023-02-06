import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # CONVOLUTION BLOCK 1
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 30
        
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 28
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 26

        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), dilation=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 22

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1 ,  padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 20

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(3, 3), stride=1 ,  padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 18

        # Dilated Convolution
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), dilation=2, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU()
        ) # output_size = 14

        # CONVOLUTION BLOCK 3
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels = 16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        ) # output_size = 14

        # CONVOLUTION BLOCK 2
        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 14
       
        # Dilated Convolution
        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), dilation=2, padding=0, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU()
        ) # output_size = 10

        # CONVOLUTION BLOCK 4   -- Depthwise Separable Convolution
        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), dilation=1, groups=64, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        ) # output_size = 8

        self.convblock12 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=10, kernel_size=(1,1), dilation=1, padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.ReLU()
        ) # output_size = 8
       
       
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) # output_size = 1

        self.dropout = nn.Dropout(0.1)         

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x=  self.dropout(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)
        x=  self.dropout(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.convblock10(x)
        x=  self.dropout(x)
        x = self.convblock11(x)
        x = self.convblock12(x)
        x = self.gap(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1) 