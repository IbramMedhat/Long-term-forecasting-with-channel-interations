import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Time Series Mixer model for modeling 
    channel & time steps interactions
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_blocks = configs.num_blocks
        self.mixer_block = MixerBlock(configs.enc_in, configs.hidden_size, configs.seq_len, configs.dropout_factor)
        self.channels = configs.enc_in
        self.pred_len = configs.pred_len 
        #Individual layer for each variate(if true) otherwise, one shared linear
        self.individual_linear_layers = configs.individual
        if(self.individual_linear_layers) :
            self.output_linear_layers = nn.ModuleList()
            for _ in range(self.channels):
                self.output_linear_layers.append(nn.Linear(configs.seq_len, configs.pred_len))
        else :
            self.output_linear_layers = nn.Linear(configs.seq_len, configs.pred_len)
            
    def forward(self, x):
        # x: [Batch, Input length, Channel]
        for _ in range(self.num_blocks):
            x = self.mixer_block(x)
        #Final linear layer applied on the transpoed mixers' output
        x = torch.swapaxes(x, 1, 2)
        # #Preparing tensor output with the correct prediction length
        y = torch.zeros([x.size(0), x.size(1), self.pred_len],dtype=x.dtype).to(x.device)
        for c in range(self.channels): 
            if self.individual_linear_layers :
                y[:, c, :] = self.output_linear_layers[c](x[:, c, :].clone())
            else :
                y[:, c, :] = self.output_linear_layers(x[:, c, :].clone())

        y = torch.swapaxes(y, 1, 2)
        return y

    
class MlpBlockFeatures(nn.Module):
    """MLP for features with 2 layer"""
    def __init__(self, channels, mlp_dim, dropout_factor):
        super(MlpBlockFeatures, self).__init__()
        self.normalization_layer = nn.BatchNorm1d(channels)
        self.linear_layer1 = nn.Linear(channels, mlp_dim)
        self.activation_layer = nn.ReLU()
        self.linear_layer2 = nn.Linear(mlp_dim, channels)
        self.dropout_layer = nn.Dropout(dropout_factor)

    def forward(self, x) :
        y = self.normalization_layer(x)
        y = self.linear_layer1(y)
        y = self.activation_layer(y)
        y = self.linear_layer2(y)
        y = self.dropout_layer(y)
        return x + y
    
class MlpBlockTimesteps(nn.Module):
    """MLP for timesteps with 1 layer"""
    def __init__(self, seq_len):
        super(MlpBlockTimesteps, self).__init__()

        self.normalization_layer = nn.BatchNorm1d(seq_len)
        self.linear_layer = nn.Linear(seq_len, seq_len)
        self.activation_layer = nn.ReLU()

    def forward(self, x) :
        y = self.normalization_layer(x)
        y = self.linear_layer(y)
        y = self.activation_layer(y)
        return x + y
    
class MixerBlock(nn.Module):
    """Mixer block layer"""
    def __init__(self, channels, features_block_mlp_dims, seq_len, dropout_factor) :
        super(MixerBlock, self).__init__()
        self.channels = channels
        self.seq_len = seq_len
        #Timesteps mixing block 
        self.timesteps_mixer = MlpBlockTimesteps(seq_len)
        #Features mixing block 
        self.channels_mixer = MlpBlockFeatures(channels, features_block_mlp_dims, dropout_factor)
    
    def forward(self, x) :
        y = torch.swapaxes(x, 1, 2)
        for c in range(self.channels):
            y[:, c, :] = self.timesteps_mixer(y[:, c, :].clone())    
        y = torch.swapaxes(y, 1, 2)
        for t in range(self.seq_len):
            y[:, t, :] = self.channels_mixer(y[:, t, :].clone()) 
        return y