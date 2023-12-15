import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, first=False):
        super(ResBlock, self).__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm1d(in_channels)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)
        
        self.conv1 = nn.Conv1d(in_channels=in_channels,
			out_channels=out_channels,
			kernel_size=3,
			padding=1,
			stride=1)
        
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
			out_channels=out_channels,
			padding=1,
			kernel_size=3,
			stride=1)
        
        if in_channels != out_channels:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = in_channels,
				out_channels = out_channels,
				padding = 0,
				kernel_size = 1,
				stride = 1)
            
        else:
            self.downsample = False

        self.mp = nn.MaxPool1d(3)

        self.fc_att = nn.Linear(in_features=out_channels, out_features=out_channels)
        self.sig = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        
    def forward(self, x):
        y = x

        if not self.first:
            x = self.bn1(x)
            x = self.lrelu(x)

        x = self.conv1(x)
        x = self.bn2(x)
        x = self.lrelu(x)
        x = self.conv2(x)

        if self.downsample:
            y = self.conv_downsample(y)
            
        x += y

        x = self.mp(x)

        temp = self.avgpool(x).view(x.size(0), -1)
        temp = self.fc_att(temp)
        temp = self.sig(temp).view(temp.size(0), temp.size(1), -1)
        x = x * temp + temp


        return x