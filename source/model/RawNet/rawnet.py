import torch
import torch.nn as nn
import torch.nn.functional as F


from .resblock import ResBlock
from .sinc_conv import SincConv_fast


class RawNet(nn.Module):
    def __init__(self, cfg):
        super(RawNet, self).__init__()

        self.Sinc_conv=SincConv_fast(out_channels=cfg["sinc_out_channels"], kernel_size=cfg["sinc_kernel_size"])
        self.bn = nn.BatchNorm1d(num_features=cfg["ResBlocks"][0][0])
        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.res_block1 = nn.Sequential(ResBlock(in_channels=cfg["ResBlocks"][0][0],
                                                 out_channels=cfg["ResBlocks"][0][1], first=True))
        self.res_block2 = nn.Sequential(ResBlock(in_channels=cfg["ResBlocks"][1][0],
                                                 out_channels=cfg["ResBlocks"][1][1]))

        self.res_block3 = nn.Sequential(ResBlock(in_channels=cfg["ResBlocks"][2][0],
                                                 out_channels=cfg["ResBlocks"][2][1]))
        self.res_block4 = nn.Sequential(ResBlock(in_channels=cfg["ResBlocks"][3][0],
                                                 out_channels=cfg["ResBlocks"][3][1]))
        self.res_block5 = nn.Sequential(ResBlock(in_channels=cfg["ResBlocks"][4][0],
                                                 out_channels=cfg["ResBlocks"][4][1]))
        self.res_block6 = nn.Sequential(ResBlock(in_channels=cfg["ResBlocks"][5][0],
                                                 out_channels=cfg["ResBlocks"][5][1]))

        
        self.bn_before_gru = nn.BatchNorm1d(num_features=cfg["ResBlocks"][5][1])
    
        self.gru = nn.GRU(input_size=cfg["gru_input_size"],
			hidden_size=cfg["gru_hidden_size"],
			num_layers=cfg["gru_num_layers"],
			batch_first=True)

        
        self.fc = nn.Linear(in_features=cfg["gru_hidden_size"], out_features=2)
       
        
        
        
    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1])
        
        x = self.Sinc_conv(x)

        x = torch.abs(x)

        x = F.max_pool1d(x, 3)
        x = self.bn(x)
        x = self.lrelu(x)
        
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        
        #check hint
        #x = self.bn_before_gru(x)
        #x = self.lrelu(x)
        x = x.permute(0, 2, 1)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:,-1,:]
        x = self.fc(x)

        return x