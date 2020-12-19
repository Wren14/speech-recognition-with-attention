import numpy as np
import json
import sys
import os 
import configargparse
import random
import numpy as np
from models.las_model import SpeechLAS
from loader import create_loader
import torch 
import argparse 


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def get_parser(parser=None,required=True):
    
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Train an automatic speech recognition (ASR) model on one CPU, "
            "one or multiple GPUs",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add("--config", is_config_file=True, help="config file path")    
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
        "--recog-json",
        default="asr_data/test/data.json",
        type=str,
        help="Recognition Features"
    )
    parser.add_argument(
        "--model-dir",
        default="exp/train_test6",
        type=str,
        help="Trained Model Directory"
    )
    parser.add_argument(
        "--decode-tag",
        default="decode_test",
        type=str,
        help="Decoding tag"
    )
    parser.add_argument(
        "--minlenratio",
        default=0.1,
        type=float,
        help="Minimum Decoding Length Ratio of Hidden State Size"
    )
    parser.add_argument(
        "--beam_size",
        default=30,
        type=int,
        help="Beam Size for Beam Decoding"
    )
    parser.add_argument(
        "--maxlenratio",
        default=0.6,
        type=float,
        help="Maximum Decoding Length Ratio of Hidden State Size"
    )
    
    return parser


def main(cmd_args):
    ## Return the arguments from parser
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)
    with open(args.recog_json, "rb") as f:
        recog_json = json.load(f)["utts"]
    
    with open(os.path.join(args.model_dir,"model.json"),"rb") as f:
        train_params = json.load(f)
    train_params = argparse.Namespace(**train_params)
    model = SpeechLAS(train_params)
    ## Load the model 
    checkpoint = torch.load(os.path.join(args.model_dir,"models","model.acc.best"))
    model.load_state_dict(checkpoint['model_state_dict'])
    if torch.cuda.is_available():
        model = model.cuda()
    
    ## Create decoding loader
    test_dataset,test_sampler,_ = create_loader(recog_json,train_params) 
    
    ## Start Decoding
    output_dict = {}
    with torch.no_grad():
        model.eval()
        for i,(feats,feat_lens,target,target_lens,test_keys) in enumerate(test_sampler):
            pred_outputs = model.decode_greedy(feats,feat_lens,args)
            for key,output in zip(test_keys,pred_outputs):
                output_string = "".join([train_params.char_list[x] for x in output]).replace("<space>"," ").replace("<eos>","")
                output_dict[key] = output_string

    if not os.path.isdir(os.path.join(args.model_dir,args.decode_tag)):
        os.mkdir(os.path.join(args.model_dir,args.decode_tag))
    output_file = os.path.join(args.model_dir,args.decode_tag,"decoded_hyp.txt")
    
    with open(output_file,"w") as f:
        f.write("\n".join(["{} {}".format(key,value) for key,value in output_dict.items()]))
        



    
    
    
    
    

if __name__ == "__main__":
    main(sys.argv[1:]) 