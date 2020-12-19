import numpy as np
import math
import torch 
import random
import torch.nn.functional as F 
from torch.nn import Embedding, LSTMCell, GRUCell, Linear
from torch.nn import Dropout,LogSoftmax
from torch.nn import ModuleList
from utils import pad_list,to_device
import configargparse
from torch.nn.functional import log_softmax


class Speller(torch.nn.Module):
    def __init__(self,params:configargparse.Namespace,att: torch.nn.Module=None):
        """
        Neural Network Module for the Sequence to Sequence LAS Model
        :params configargparse.Namespace params: The training options
        :params torch.nn.Module att: The attention module
        """        
        super(Speller,self).__init__()
        ## Embedding Layer
        self.embed = Embedding(params.odim,params.demb_dim)
        ## Decoder with LSTM Cells
        self.decoder = ModuleList()
        self.dropout_dec = ModuleList()
        self.dtype = params.dtype
        self.dunits = params.dhiddens
        self.dlayers = params.dlayers
        self.decoder += [
                LSTMCell(params.eprojs + params.demb_dim, params.dhiddens)
                if self.dtype == "lstm"
                else GRUCell(params.eprojs + params.demb_dim, params.dhiddens)
            ]
        self.dropout_dec += [Dropout(p=params.ddropout)]
        self.dropout_emb = Dropout(p=params.ddropout)
        ## Other decoder layers if > 1 decoder layer
        for i in range(1,params.dlayers): 
            self.decoder += [
                LSTMCell(params.dhiddens, params.dhiddens)
                if self.dtype == "lstm"
                else GRUCell(params.dhiddens, params.dhiddens)
            ]
            self.dropout_dec += [Dropout(p=params.ddropout)]
        
        ## Project to Softmax Space- Output
        self.projections = Linear(params.dhiddens, params.odim)
        self.softmax = LogSoftmax(dim=-1)
        ## Attention Module
        self.att = att
        ## Scheduled Sampling
        self.sampling_probability = params.ssprob
        ## Initialize EOS, SOS
        self.eos = len(params.char_list) -1
        self.sos = self.eos
        self.ignore_id = params.text_pad

    def rnn_forward(self, lstm_input: torch.Tensor, dec_hidden_states: list, dec_hidden_cells: list, dec_hidden_states_prev: list, dec_hidden_cells_prev: list):
        """
        Performs forward pass through LSTMCells in the decoder
        :param torch.Tensor lstm_input- concatenated embedding vector and attention context that is input to first LSTMCell layer
        :param list(torch.Tensor) dec_hidden_states- Hidden states of the decoder for all layers in a list.
        :param list(torch.Tensor) dec_hidden_cells- Hidden cells of the decoder for all layers in a list.
        :param list(torch.Tensor) dec_hidden_states_prev- Hidden states of the decoder for all layers in a list.
        :param list(torch.Tensor) dec_hidden_states_cells- Hidden cells of the decoder for all layers in a list.
        :returns list(torch.Tensor) dec_hidden_states- Hidden states of the decoder for all layers in a list.
        :returns list(torch.Tensor) dec_hidden_cells- Hidden cells of the decoder for all layers in a list.
        """

        dec_hidden_states[0], dec_hidden_cells[0] = self.decoder[0](lstm_input, (dec_hidden_states_prev[0], dec_hidden_cells_prev[0]))
        for i in range(1, self.dlayers):
            dec_hidden_states[i], dec_hidden_cells[i] = self.decoder[i](
                self.dropout_dec[i - 1](dec_hidden_states[i - 1]), (dec_hidden_states_prev[i], dec_hidden_cells_prev[i])
            )
        return dec_hidden_states, dec_hidden_cells
    
    def forward(self,hs:torch.Tensor,hlens:list,ys_out:torch.LongTensor,ylen:list):
        """
        Performs the forward pass over the decoder 
        :param torch.Tensor hs- Encoded output sequence
        :param list hlens- Lengths of the encoded output sequence without padding
        :param torch.LongTensor ys_out- Target output sequence with padding 
        :param list ylen- Target sequence lengths without padding
        :returns torch.Tensor logits: Output projection to vocabulary space [Batch,Max_token_length,Vocabulary_size] 
        :returns torch.LongTensor- Target output sequence with <eos> added to the end for loss computation
        """
        
        hlens = hlens.tolist()
        self.att.reset()
        max_len = ys_out.size(1) + 1
        logits_list = []

        # Initialization
        attw = None 
        hidden_state_list = [hs.new_zeros(hs.size(0), self.dunits) for layer in range(self.dlayers)]
        hidden_cell_list = [hs.new_zeros(hs.size(0), self.dunits) for layer in range(self.dlayers)]
         
        hidden_cell_list_prev = [hs.new_zeros(hs.size(0), self.dunits) for layer in range(self.dlayers)]
        hidden_state_list_prev = [hs.new_zeros(hs.size(0), self.dunits) for layer in range(self.dlayers)]
       


        ys_unpadded = [ ys[ys != -1] for ys in ys_out ]
        eos = ys_unpadded[0].new([self.eos])
        sos = ys_unpadded[0].new([self.sos])
        ys_charemb = [ torch.cat((sos,ys), dim=0) for ys in ys_unpadded]
        ys_charemb_padded = pad_list(ys_charemb,self.eos)
        ys_ref = [torch.cat((ys,eos), dim=0) for ys in ys_unpadded]
        #ys_ref_padded = pad_list(ys_ref,-1)
        ys_ref_padded = pad_list(ys_ref, self.eos)
        prev_time = hs.new_tensor([self.sos for batch in range(hs.size(0))]).long()
        # Compute Embeddings
        embedd = self.embed(ys_charemb_padded)

        threshold = 0.2
        for i in range(max_len):
            # Location Attention
            attc, attw =  self.att(hs, hlens, hidden_state_list[0], attw)
            # Concatenate embedding and attention
            #if epoch < 10 random.random < 0.1
            # Scheduled sampling :-> Use --resume to train with different threshold set
            if random.random() < threshold:
                new_embed = self.embed(prev_time)
                Linp_i = torch.cat((attc, new_embed), dim=1)
            else:
                Linp_i = torch.cat((attc, embedd[:, i-1, :]), dim=1)
            

            # RNN Forward
            hidden_state_list, hidden_cell_list = self.rnn_forward(Linp_i, hidden_state_list, hidden_cell_list, hidden_state_list_prev, hidden_cell_list_prev)
            hidden_cell_list_prev = hidden_cell_list
            hidden_state_list_prev = hidden_state_list

            # Projections
            Li_classes = self.projections(hidden_cell_list[-1])
            topv, topi = Li_classes.topk(1, dim=-1)
            prev_time = topi.squeeze().detach() 
            logits_list.append(Li_classes)
            
        logits_list = torch.stack(logits_list, dim=1)
        return logits_list, ys_ref_padded 
       #raise NotImplementedError
             

    def greedy_decode(self,hs:torch.Tensor,hlens:torch.LongTensor,params:configargparse.Namespace):
        """
        Performs greedy decoding  
        :param torch.Tensor hs- Encoded output sequence
        :param list hlens- Lengths of the encoded output sequence without padding
        :param Namespace params- Decoding options
        """
        #print(f"SOS Token : {self.sos} EOS Token : {self.eos}")
        hidden_state_list = [hs.new_zeros(hs.size(0), self.dunits) for layer in range(self.dlayers)]
        hidden_cell_list = [hs.new_zeros(hs.size(0), self.dunits) for layer in range(self.dlayers)]
        hidden_cell_list_prev = [hs.new_zeros(hs.size(0), self.dunits) for layer in range(self.dlayers)]
        hidden_state_list_prev = [hs.new_zeros(hs.size(0), self.dunits) for layer in range(self.dlayers)]
        self.att.reset()
        hlens = hlens.tolist()
        attw = None
        max_len = 200
        decoded_words = torch.tensor([]).long()
        ys = hs.new_tensor([self.sos for batch in range(hs.size(0))]).long()
        
        for i in range(max_len):
            embedd = self.embed(ys)
            attc, attw =  self.att(hs, hlens, hidden_state_list[0], attw)
            
            Linp_i = torch.cat((attc, embedd), dim=1)
            
            hidden_state_list, hidden_cell_list = self.rnn_forward(Linp_i, hidden_state_list, hidden_cell_list, hidden_state_list_prev, hidden_cell_list_prev)
            hidden_cell_list_prev = hidden_cell_list
            hidden_state_list_prev = hidden_state_list
            Li_classes = self.projections(hidden_cell_list[-1])
            raw_pred = log_softmax(Li_classes, dim=-1)

            topv, topi = raw_pred.topk(1, dim=-1) # N*1 dimension

            if len(decoded_words) == 0:
                decoded_words = topi
            else:
                decoded_words = torch.cat((decoded_words, topi), dim=-1)
            ys = topi.squeeze().detach()

        # Chop off characters after eos
        decoded_words = decoded_words.tolist()
        for lst in decoded_words:
            if self.eos in lst:
                ind = lst.index(self.eos)
                lst[:] = lst[:ind+1]
        return decoded_words
            

    
       #raise NotImplementedError