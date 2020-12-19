#! /bin/bash

import numpy as np
import json
import sys
import os 
import configargparse
import random
import numpy as np
from trainer import Trainer

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def get_parser(parser=None,required=True):
    
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Train an automatic speech recognition (ASR) model on one CPU, "
            "one or multiple GPUs",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add("--config", is_config_file=True, help="config file path",default="conf/base.yaml")    
    ## General utils
    
    parser.add_argument(
        "--tag", 
        type=str,
        help="Experiment Tag for storing logs,models"
    )
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Random seed"
    )
    parser.add_argument(
        "--text_pad",
        default=-1,
        type=int,
        help="Padding Index for Text Labels"
    )
    parser.add_argument(
        "--audio_pad",
        default=0,
        type=int,
        help="Padding Index for Audio features"
    )



    ## I/O parameters
   
    parser.add_argument(
        "--train-json",
        type=str,
        default="asr_data/train_si284/data.json",
        help="Filename of train label data (json)",
    )
    parser.add_argument(
        "--valid-json",
        type=str,
        default="asr_data/test_dev93/data.json",
        help="Filename of validation label data (json)",
    )
    parser.add_argument(
        "--dict",
        type=str,
        default="asr_data/train_si284_units.txt",
        help="Filename of the dictionary file",
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="exp/base",
        help="Output Data Directory/Experiment Directory"
    )
    
    
    ## Model Parameters
    parser.add_argument(
        "--idim",
        type=int, 
        default=83,
        help="Input Feature Size"
    )
    ## Encoder
    parser.add_argument(
        "--ehiddens", 
        type=int, 
        default=320,
        help="Encoder Hidden Size"
    )
    parser.add_argument(
        "--eprojs", 
        type=int, 
        default=320,
        help="Encoder Projection Output Size"
    )
    parser.add_argument(
        "--etype", 
        type=str, 
        default="blstm",
        help="Encoder Type"
    )
    parser.add_argument(
        "--elayers",
        type=int,
        default=3,
        help="Number of encoder layers"
    )
    parser.add_argument(
        "--pfactor",
        type=int,
        default=2,
        help="Pyramidal Downsampling factor"
    )
    ## Decoder
    parser.add_argument(
        "--dtype", 
        type=str, 
        default="lstm",
        help="Decoder Type"
    )
    parser.add_argument(
        "--dhiddens",
        type=int,
        default=300,
        help="Decoder Hidden Size"
    )
    parser.add_argument(
        "--demb_dim",
        type=int,
        default=100,
        help="Decoder Embedding Size"
    )
    parser.add_argument(
        "--dlayers",
        type=int,
        default=2,
        help="Number of decoder layers"
    )
    parser.add_argument(
        "--ddropout",
        type=float,
        default=0.0,
        help="Decoder Dropout Ratio"
    )
    parser.add_argument(
        "--ssprob",
        type=float,
        default=0.0,
        help="Probability of Scheduled Sampling"
    )
    ## Attention 
    parser.add_argument(
        "--att_dim",
        type=int,
        default=320,
        help="Attention Projection Dimension for Location Based Attention"
    )
    parser.add_argument(
        "--aconv_chans",
        type=int,
        default=10,
        help="Attention Convolution Channels for Location Based Attention"
    )
    parser.add_argument(
        "--aconv_filts",
        type=int,
        default=100,
        help="Attention Convolution Filter Size for Location Based Attention"
    )

    ## Batching
    parser.add_argument(
        "--batch_bins",
        type=int,
        default=800000
    )
    parser.add_argument(
        "--nworkers",
        dest="nworkers",
        type=int,
        default=0
    )

    ## Optimization Parameters
    parser.add_argument(
        "--opt",
        default="adadelta",
        type=str,
        choices=["adadelta", "adam"],
        help="Optimizer",
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.5
    )
    parser.add_argument(
        "--opt_patience",
        type=int,
        default=5
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=5
    )
    parser.add_argument(
        "--wdecay",
        type=float,
        default=0,
        help="Weight Decay Parameter"
    )
    parser.add_argument(
        "--accum-grad",
        default=1,
        type=int,
        help="Number of gradient accumuration"
    )
    parser.add_argument(
        "--eps",
        default=1e-8,
        type=float,
        help="Epsilon constant for optimizer"
    )
    parser.add_argument(
        "--eps-decay",
        default=0.01,
        type=float,
        help="Decaying ratio of epsilon"
    )
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        nargs="?",
        help="Number of epochs to wait without improvement before stopping the training",
    )
    
    
    ## Model Initialization
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path of model and optimizer to be loaded"
    )


    ## Training Config
    # Changing from 30 to 1
    parser.add_argument("--nepochs", 
        type=int, 
        default=30,
        help="Number of GPUs used for training"
    )
    parser.add_argument(
        "--ngpu",
        default=1,
        type=int,
        help="Number of GPUs. If not given, use all visible devices",
    )
    parser.add_argument(
        "--num-batch-save-attention",
        default=1,
        type=int,
        help="Number of samples of attention to be saved",
    )


    return parser


def main(cmd_args):
    
    ## Return the arguments from parser
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    
    ## Set Random Seed for Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    ## Set all directories   
    expdir = os.path.join("exp","train_"+args.tag)
    graph_dir = os.path.join(expdir,"plots")
    model_dir = os.path.join(expdir,"models")
    log_dir = os.path.join(expdir,"logs")
    tb_dir = os.path.join("tensorboard","train_"+args.tag)
    
    args.graph_dir = graph_dir
    args.model_dir = model_dir
    args.log_dir = log_dir
    args.tb_dir = tb_dir
    args.expdir = expdir
    
    for x in ["exp","tensorboard",expdir,graph_dir,model_dir,log_dir,tb_dir]:
        if not os.path.isdir(x):
            os.mkdir(x)
    
    
    ## Load the Character List, add <unk>- index 0
    ## We use the same token to represent <eos> and <sos>- last index
    char_list = ["<unk>"]
    with open(args.dict,"r",encoding="utf-8") as f:
        char_dict = [line.strip().split(' ')[0] for line in f.readlines()]
    char_list = char_list + char_dict + ["<eos>"] 
    args.char_list = char_list
    args.odim = len(char_list)
    ## Start training
    trainer = Trainer(args)
    trainer.train()
 



if __name__ == "__main__":
    main(sys.argv[1:]) 
