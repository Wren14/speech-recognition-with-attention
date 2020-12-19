import numpy as np
import sys
import torch
from torch.nn import LSTM,RNN,GRU,Linear
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import configargparse


class RNNLayer(torch.nn.Module):
    
    def __init__(self, idim: int ,hdim: int ,nlayers: int=1, enc_type: str ="blstm", edropout=0.4):
        """
        This represents the computation that happens for 1 RNN Layer
        Uses packing,padding utils from Pytorch
        :param int input_dim- The input size of the RNN
        :param int hidden_dim- The hidden size of the RNN
        :param int nlayers- Number of RNN Layers
        :param str enc_type : Type of encoder- RNN/GRU/LSTM
        """
        super(RNNLayer,self).__init__()
        bidir = True if enc_type[0] == 'b' else False
        enc_type = enc_type[1:] if enc_type[0] == 'b' else enc_type
        if enc_type == "rnn":
            self.elayer = RNN(idim, hdim, nlayers, batch_first=True, bidirectional=bidir, dropout=edropout)
        elif enc_type == "lstm":
            self.elayer = LSTM(idim, hdim, nlayers, batch_first=True, bidirectional=bidir, dropout=edropout)
        else:
            self.elayer = GRU(idim, hdim, nlayers, batch_first=True, bidirectional=bidir, dropout=edropout)

    def forward(self, x: torch.Tensor, inp_lens: torch.LongTensor):
        """
        Foward propogation for the RNNLayer
        :params torch.Tensor x - Input Features
        :params torch.LongTensor inp_lens - Input lengths without padding
        :returns torch.Tensor Encoded output
        :returns list Encoded output lengths
        :returns Encoder hidden state
        """
        total_length = x.size(1)
        packed_x = pack_padded_sequence(x, inp_lens,batch_first=True)
        self.elayer.flatten_parameters()
        output, (hidden,_) = self.elayer(packed_x)
        unpacked_out,inp_lens = pad_packed_sequence(output,batch_first=True,total_length=total_length)
        return unpacked_out,inp_lens,hidden
        

class pBLSTM(torch.nn.Module):
    
    def __init__(self, input_dim: int , hidden_dim: int,subsample_factor:int = 2,enc_type: str ="blstm"):
        """
        Pyramidal BLSTM Layer: 
        :param int input_dim- The input size of the RNN
        :param int hidden_dim- The hidden size of the RNN
        :param int subsample_factor- Determines the factor by which the time dimension is downsampled and 
                                the hidden state is upsampled. Value 2 was used in the LAS paper
        :param str enc_type : Type of encoder- RNN/GRU/LSTM
        It takes in a sequence of shape [bs,enc_length,hiddens], 
        and converts it to a sequence of shape [bs,enc_length//subsample_factor, hidden*subsample_factor]
        """
        super(pBLSTM, self).__init__()
        self.factor = subsample_factor
        self.pblstm = RNNLayer(self.factor*input_dim,hidden_dim,1,enc_type)
        self.input_dim = input_dim
        
    def forward(self, x: torch.Tensor , inp_lens: torch.LongTensor ): 
        """
        Foward propogation for the pBLSTM
        :params torch.Tensor x - Input Features
        :params torch.LongTensor inp_lens - Input lengths without padding
        :returns torch.Tensor Encoded output
        :returns list Encoded output lengths
        :returns Encoder hidden state
        """
        batch_size,seq_max, hidden_dim = x.size()
        ## Handle seq_len % factor !=0 by dropping the last few frames
        
        # Reduce Sequence Length by factor of 2 and multiply hidden_dim
        time_reduction = int(seq_max/2)
        if seq_max%2 != 0:
            x = x[:,:seq_max-1, :]
        
        inp_lens /= 2
        input_x = x.contiguous().view(batch_size, time_reduction, hidden_dim*2)
        
        # Pass through pblstm
        unpacked_out, inp_lens, hidden = self.pblstm(input_x, inp_lens)
        return unpacked_out, inp_lens, hidden
       
        

    


class Listener(torch.nn.Module):
    def __init__(self,params:configargparse.Namespace):
        """
        Neural Network Module for the Listener
        :params configargparse.Namespace params: The training options
        """      
        super(Listener, self).__init__()
        self.elayers = params.elayers
        self.etype = params.etype
        rnn0 = RNNLayer(params.idim,params.ehiddens,1,params.etype)
        setattr(self, "%s0" % (params.etype),rnn0)
        for i in range(params.elayers):
            rnn = pBLSTM(2*params.ehiddens,params.ehiddens,params.pfactor,params.etype)
            setattr(self, "%s%d" % (params.etype, i+1), rnn)   
            if i == params.elayers-1:
                projection = Linear(2*params.ehiddens,params.eprojs)
                setattr(self,"proj%d" % (i+1),projection)
    

    def forward(self,x: torch.Tensor,inp_lens: torch.LongTensor):
        """
        Foward propogation for the encoder
        :params torch.Tensor x - Input Features
        :params torch.LongTensor inp_lens - Input lengths without padding
        :returns torch.Tensor Encoded output
        :returns list Encoded output lengths
        """
        L_prev, Llens_prev, _ = getattr(self, "%s0" % (self.etype))(x, inp_lens)
        for i in range(1, self.elayers+1):
            L_i, Llens_i, _ = getattr(self, "%s%d" % (self.etype, i))(L_prev, Llens_prev)
            L_prev, Llens_prev = L_i, Llens_i

        L_prev = getattr(self, "proj%d" % (self.elayers))(L_prev)
        return L_prev, Llens_prev 
        
        

