import torch
import torch.nn as nn
import math

class Model(nn.Module):
    """
    Patches-TSMixer based LTSF model mixing accross channels and patches
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # Patching related parameters and layers
        self.patch_size = configs.patch_size
        self.seq_len = configs.seq_len
        self.channels = configs.enc_in
        self.num_patches = math.ceil(self.seq_len / self.patch_size)
        self.remaining_timesteps = (self.num_patches * self.patch_size) - self.seq_len
        if self.remaining_timesteps == 0 :
            # No padding is needed
            self.replicationPadLayer = nn.ReplicationPad1d((0, 0))
            self.output_linear_layers = nn.Linear(configs.seq_len, configs.pred_len)

        else :
            padding_length = int(self.patch_size - self.remaining_timesteps)
            self.replicationPadLayer = nn.ReplicationPad1d((0, padding_length))
            # Adjusting the projection linear layer to accomodate the padded sequence
            self.output_linear_layers = nn.Linear(configs.seq_len+padding_length, configs.pred_len)

        # Mixing related parameters and layers
        self.mixer_block = MixerBlock(self.channels, self.num_patches, self.patch_size, configs.activation, configs.dropout)
        self.pred_len = configs.pred_len
        self.num_blocks = configs.num_blocks


    def forward(self, x):
        # Reshaping to [Batches, Channels, Timesteps]
        x = torch.swapaxes(x, 1, 2)
        # Repeating last values to make the number of timesteps divisible by the number of patches
        x = self.replicationPadLayer(x)
        # Reshaping to [Batches, Num_patches(seq_len/patch_size), Patch_size, Channels]
        num_patches = x.size(2) // self.patch_size
        x = torch.reshape(x, 
                          (x.size(0),
                            num_patches,
                            self.patch_size,
                            x.size(1)))
        # Applying mixing step (keeps diminsions as is)
        for _ in range(self.num_blocks) :
            x = self.mixer_block(x)
        # Reshaping to [Batch, channels, padded_sequance]
        x = torch.reshape(x, 
                          (x.size(0),
                            self.channels,
                            num_patches*self.patch_size))
        # Preparing tensor output with the correct prediction length
        # Output tensor shape of [Batch, channel, pred_len]
        y = torch.zeros([x.size(0), self.channels, self.pred_len],dtype=x.dtype).to(x.device)
        y = self.output_linear_layers(x.clone())
        y = torch.swapaxes(y, 1, 2)
        return y
        
class MlpBlockFeatures(nn.Module):
    """MLP for features"""
    def __init__(self, channels, activation, dropout_factor):
        super(MlpBlockFeatures, self).__init__()
        self.normalization_layer = nn.BatchNorm2d(channels)
        self.linear_layer = nn.Linear(channels, channels)
        self.dropout_layer = nn.Dropout(dropout_factor)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None

    def forward(self, x) :
        # Swapping channel diminsion for applying normalization first
        normalized_input = torch.swapaxes(x, 1, 3)
        normalized_input = self.normalization_layer(normalized_input)
        normalized_input = torch.swapaxes(normalized_input, 1, 3)
        y = self.linear_layer(normalized_input)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)      
        return normalized_input + y
    
class MlpBlockPatches(nn.Module):
    """MLP for patches"""
    def __init__(self, num_patches, activation, dropout_factor):
        super(MlpBlockPatches, self).__init__()
        self.normalization_layer = nn.BatchNorm2d(num_patches)
        self.linear_layer = nn.Linear(num_patches, num_patches)
        self.dropout_layer = nn.Dropout(dropout_factor)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None


    def forward(self, x) :
        # [batch_size, num_patches, patch_size, channels]
        normalized_input = self.normalization_layer(x)
        y = torch.swapaxes(normalized_input, 1, 3)
        y = self.linear_layer(y)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)
        y = torch.swapaxes(y, 1, 3)
        return normalized_input + y    
    
class MlpBlockPatchSize(nn.Module):
    """MLP for num_patches"""
    def __init__(self, patch_size, activation, dropout_factor):
        super(MlpBlockPatchSize, self).__init__()
        self.normalization_layer = nn.BatchNorm2d(patch_size)
        self.linear_layer = nn.Linear(patch_size, patch_size)
        self.dropout_layer = nn.Dropout(dropout_factor)
        if activation=="gelu" :
            self.activation_layer = nn.GELU()
        elif activation=="relu" :
            self.activation_layer = nn.ReLU()
        else :
            self.activation_layer = None


    def forward(self, x) :
        # [batch_size, num_patches, patch_size, channels]
        normalized_input = torch.swapaxes(x, 1, 2)
        # [batch_size, patch_size, num_patches, channels]
        normalized_input = self.normalization_layer(normalized_input)
        y = torch.swapaxes(normalized_input, 1, 3) # to make the patch_size the last diminsion for the mixing process
        # [batch_size, channels, num_patches, patch_size]
        y = self.linear_layer(y)
        if self.activation_layer is not None :
            y = self.activation_layer(y)
        y = self.dropout_layer(y)

        # Reshaping to original diminsions for residual connection and further computation
        y = torch.swapaxes(y, 1, 3)
        y = torch.swapaxes(y, 1, 2)
        normalized_input = torch.swapaxes(normalized_input, 1, 2)
        # [batch_size, num_patches, patch_size, channels]
        return normalized_input + y  

class MixerBlock(nn.Module):
    """Mixer block layer only mixing channels in this model"""
    def __init__(self, channels, num_patches, patch_size, activation, dropout_factor) :
        super(MixerBlock, self).__init__()
        self.channels_mixer = MlpBlockFeatures(channels, activation, dropout_factor)
        self.patches_mixer = MlpBlockPatches(num_patches, activation, dropout_factor)
        self.patchSize_mixer = MlpBlockPatchSize(patch_size, activation, dropout_factor)

    def forward(self, x) :
        y = self.channels_mixer(x)
        y = self.patches_mixer(y)
        y = self.patchSize_mixer(y)
        return y

