import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Patches-Conv based LTSF model
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.channels = configs.enc_in
        self.kernel_size = configs.kernel_size
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.stride = configs.stride
        self.normalization_layer_channels = nn.BatchNorm1d(self.channels)
        self.normalization_layer_seq_len = nn.BatchNorm1d(self.seq_len)

        self.conv_layer_time_mixing = nn.Conv1d(self.seq_len, self.seq_len, self.kernel_size,
                                                stride=self.stride, padding_mode='replicate')                                                               
        self.conv_layer_channel_mixing =  nn.Conv1d(self.channels, self.channels, self.kernel_size,
                                      stride=self.stride, padding_mode='replicate')
        
        if configs.activation=="gelu" :
            self.activation_layer_time_mixing = nn.GELU()
            self.activation_layer_channel_mixing = nn.GELU()
        elif configs.activation=="relu" :
            self.activation_layer_time_mixing = nn.ReLU()
            self.activation_layer_channel_mixing = nn.ReLU()
        else :
            self.activation_layer_time_mixing = None
            self.activation_layer_channel_mixing = None
        
        self.linear_output_layer = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        # Implementing time mixing through conv
        y = self.normalization_layer_seq_len(x)
        y = self.conv_layer_time_mixing(y)
        y = self.activation_layer_time_mixing(y)
        y = x + y #Residual Connection
        y = torch.swapaxes(y, 1, 2)
        # Implementing channel mixing through conv
        y2 = self.normalization_layer_channels(y)
        y2 = self.conv_layer_channel_mixing(y2)
        y2 = self.activation_layer_channel_mixing(y2)
        y2 = y2 + y #Residual Connection
        output = torch.zeros([x.size(0), self.channels, self.pred_len],dtype=x.dtype).to(x.device)
        output = self.linear_output_layer(y2.clone())
        output = torch.swapaxes(output, 1, 2)
        return output
