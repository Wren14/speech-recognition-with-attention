Sequence to Sequence Speech Recognition System based on [Listen Attend and Spell](https://arxiv.org/abs/1508.01211). The encoder uses pyramidal bi-LSTM (pBLSTM) which obtains 
hidden representations from the input audio features and the decoder produces language tokens in an autoregressive fashion.
Attention is used to obtain alignments between the encoded outputs and the decoder outputs at every decoding time step.
