import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
import os
from models.las_model import SpeechLAS
import math
import torch 
from torch.nn import DataParallel
from torch.optim import Adam,Adadelta
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from itertools import chain
## Data Loading Utilities
from loader import create_loader
import time
import configargparse

class Trainer:
    def __init__(self,params: configargparse.Namespace):
        """
        Initializes the Trainer with the training parameters provided in train.py
        """
        self.params = params
        self.nepochs = params.nepochs
        self.ngpu = params.ngpu
        with open(params.train_json, "rb") as f:
            train_json = json.load(f)["utts"]
        with open(params.valid_json, "rb") as f:
            valid_json = json.load(f)["utts"]
        torch.backends.cudnn.benchmark=True

        self.train_dataset,self.train_sampler,batch_list = create_loader(train_json,params)        
        self.valid_dataset,self.valid_sampler,_ = create_loader(valid_json,params)

        ## Build Model
        self.model = SpeechLAS(params)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        if self.ngpu > 1:
            self.model = DataParallel(self.model) 
        
        print(self.model)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        params.tparams = total_params
        print("Built a model with {:2.2f}M Params".format(float(total_params)/1000000))

        ## Write out model config
        with open(os.path.join(params.expdir,"model.json"),"wb") as f:
            f.write(json.dumps(vars(params), indent=4, sort_keys=True).encode('utf_8'))    


        ## Optimizer
        if params.opt == "adadelta":
            self.opt = Adadelta(self.model.parameters(),lr=params.lr,eps=params.eps,weight_decay=params.wdecay)
        else:
            self.opt = Adam(self.model.parameters(),lr=params.lr,weight_decay=params.wdecay)
        self.opt_patience = 0        
        
        ## Initialize Stats for Logging
        self.train_stats = {}
        self.val_stats = {}
        self.val_stats["best_acc"] = 0.0
        self.val_stats["best_epoch"] = 0
        self.writer = SummaryWriter(self.params.tb_dir)
        
        ## Resume/Load Model
        if params.resume != '':
            self.resume_training(params.resume)
        else:
            self.epoch = 0 
        self.start_time = time.time()
    
    def train(self):
        """
        Performs ASR Training using the provided configuration.
        This is the main training wrapper that trains and evaluates the model across epochs

        """   
        
        while self.epoch < self.nepochs:
            ## Reset Model Stats
            self.reset_stats()
            start_time = time.time()
            ## Train 1 Epoch
            self.train_epoch()
            ## validate 1 Epoch
            self.validate_epoch()
            end_time = time.time()
            ## Log Tensorboard and LogFile
            print("Epoch {}| Training: Loss {:.2f} , Accuracy {:.2f}| Validation: Loss {:.2f} Accuracy {:.2f} WER {:.2f}| Time: this Epoch {:.2f}s Elapsed {:.2f}s\n\n".format(self.epoch,self.train_stats["loss"],self.train_stats["acc"],self.val_stats["loss"],self.val_stats["acc"],self.val_stats["wer"],end_time-start_time,end_time-self.start_time))
            self.log_epoch()
            ## Save Models 
            self.save_model()
            self.epoch += 1
        
    def train_epoch(self):
        """"
        Contains the training loop across all training data to update the model in an epoch
        """
        ## Training Pass
        self.model.train()
        nsamps = 0
        for i,(feats,feat_lens,target,target_lens,train_keys) in enumerate(self.train_sampler):   
            self.opt.zero_grad()
            loss,acc,_ = self.model(feats,feat_lens,target,target_lens)
            loss.backward()
            ## Set Statistics for this Batch Update
            self.train_stats["nbatches"] +=1
            nsamps += len(train_keys)
            self.train_stats["acc"] += acc
            self.train_stats["loss"] += loss.item()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.params.grad_clip)
            if math.isnan(grad_norm):
                print('grad norm is nan. Do not update model.')
            else:
                self.opt.step()
        ## For last batch, we plot out the gradients through the network
        self.gradient_plot()        
        ## Obtain Utterance Level Parameters as opposed to Batch Parameters 
        self.train_stats["acc"] /= self.train_stats["nbatches"]
        self.train_stats["loss"] /=  self.train_stats["nbatches"]
        
    def validate_epoch(self):
        """"
        Contains the validation loop across all validation data to update the model in an epoch
        """
        ## Validation Pass
        self.model.eval()
        nsamps = 0
        with torch.no_grad():
            for i,(feats,feat_lens,target,target_lens,valid_keys) in enumerate(self.valid_sampler):   
                loss,acc,wer = self.model(feats,feat_lens,target,target_lens)
                if i ==0:
                    print("Example Decoding: \nREF {} \n HYP {}".format(self.model.stat_calculator.ref,self.model.stat_calculator.hyp))

                nsamps += len(valid_keys)
                self.val_stats["nbatches"] +=1
                self.val_stats["acc"] += acc
                self.val_stats["wer"] += wer
                self.val_stats["loss"] += loss.item()
                if self.valid_sampler.__len__() -1 -i <= self.params.num_batch_save_attention:
                    ## Retrieve Attention Weights 
                    att_ws = self.model.decoder.att.att_wt.detach().cpu().numpy()
                    pred_tokens = self.model.stat_calculator.pred_tokens
                    self.plot_attentions(att_ws,valid_keys,pred_tokens)
            self.val_stats["acc"] /= self.val_stats["nbatches"]
            self.schedule_optimizer_decay()
            self.val_stats["loss"] /=  self.val_stats["nbatches"]
            self.val_stats["wer"] /= self.val_stats["nbatches"]
            print("Validated over {} utterances in total".format(nsamps))

    
        
    def resume_training(self,path: str ):
        """
        Utility function to load a previous model and optimizer checkpoint, and set the starting epoch for resuming training
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.val_stats["best_epoch"] = checkpoint['epoch']
        self.val_stats["best_acc"] = checkpoint['acc']
    
    
    def reset_stats(self):
        """
        Utility function to reset training and validation statistics at the start of each epoch
        """
        self.train_stats["nbatches"] = 0
        self.train_stats["acc"] = 0
        self.train_stats["loss"] = 0
        self.val_stats["nbatches"] = 0
        self.val_stats["acc"] = 0
        self.val_stats["loss"] = 0
        self.val_stats["wer"] = 0
    
    
    def save_model(self):
        """
        Utility function to save the model snapshot after every epoch of training. 
        Saves the model after each epoch as <model-path>/snapshot.ep{}.pth
        Saves the model with highest validation accuracy thus far (and least CER) as <model-path>/model.acc.best
        Updates the best validation accuracy and epoch with the best validation accuracy in validation stats dictionary
        """
        torch.save(
            {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'acc': self.val_stats["acc"],
            }, os.path.join(self.params.model_dir,"snapshot.ep{}.pth".format(self.epoch)))
        if self.val_stats["best_acc"] <= self.val_stats["acc"]:
            self.val_stats["best_acc"] = self.val_stats["acc"]
            self.val_stats["best_epoch"] = self.epoch
            print("Saving model after epoch {}".format(self.epoch))
            torch.save(
                {
                'epoch': self.epoch,
                'model_state_dict': self.model.state_dict(),
                'acc': self.val_stats["acc"],
                }, os.path.join(self.params.model_dir,"model.acc.best"))
        #else:
        #    checkpoint = torch.load(os.path.join(self.params.model_dir,"model.acc.best"))
        #    self.model.load_state_dict(checkpoint["model_state_dict"])

    def schedule_optimizer_decay(self):
        if self.val_stats["best_acc"] > self.val_stats["acc"]:
            self.opt_patience += 1
        if self.opt_patience == self.params.opt_patience:
            if self.params.opt == "adadelta":
                for p in self.opt.param_groups:
                    p["eps"] *= self.params.eps_decay
                    print("Decayed Optimizer Eps to {}".format(p["eps"]))
            else:
                for p in self.opt.param_groups:
                    p["lr"] *= self.params.lr_decay
                    print("Decayed Optimizer LR to {}".format(p["lr"]))
            self.opt_patience = 0

    def log_epoch(self):
        """
        Utility function to write parameters from the Training and Validation Statistics Dictionaries onto Tensorboard at the end of each epoch
        """
        self.writer.add_scalar("training/acc",self.train_stats["acc"],self.epoch)
        self.writer.add_scalar("training/loss",self.train_stats["loss"],self.epoch)
        self.writer.add_scalar("validation/acc",self.val_stats["acc"],self.epoch)
        self.writer.add_scalar("validation/loss",self.val_stats["loss"],self.epoch)
        self.writer.add_scalar("validation/best_acc",self.val_stats["best_acc"],self.epoch)
    
    
    def plot_attentions(self,att_wts:np.array,keys:list,pred_tokens:list=None): 
        """
        Plots the attention weights as a figure and on tensorboard for a batch of validation examples
        :param np.array att_wts - Attention weights for a batch of inputs
        :param list(str) keys- List of utterance ID keys
        :param list(list) pred_tokens- List of lists containing the predicted tokens
        """
        import matplotlib
        from matplotlib.ticker import MaxNLocator

        for i,(key,att_w) in enumerate(zip(keys,att_wts)):
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            plt.clf()
            fig = plt.figure(figsize=(40,30))
            ax = plt.axes()
            if pred_tokens is not None:
                ytokens = [self.params.char_list[x] for x in pred_tokens[i]]
            att_w = att_w.astype(np.float32)[:len(ytokens)+1,:]
            plt.imshow(att_w, aspect="auto")
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if pred_tokens is not None:
                ax.set_yticks(np.linspace(0, len(ytokens) - 1, len(ytokens)))
                ax.set_yticks(np.linspace(0, len(ytokens) - 1, 1), minor=True)
                ax.set_yticklabels(ytokens)
            plt.tight_layout()
            plt.savefig(os.path.join(self.params.graph_dir,"ep{}-{}.png".format(self.epoch,key)))
            self.writer.add_figure(key,fig,self.epoch)
            
            
    def gradient_plot(self):
        """
        Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.
        
        This is called after loss.backwards() to visualize the gradient flow

        """
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in self.model.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                if p.grad is None:
                    print("Error n={} has None grads".format(n))
                layers.append(n)
                ave_grads.append(p.grad.abs().mean())
                max_grads.append(p.grad.abs().max())
        fig = plt.figure()
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        plt.savefig(self.params.graph_dir+"gradflow_%d.png" % (self.epoch),bbox_inches='tight')
        self.writer.add_figure("grad_flow",fig,self.epoch)
        plt.clf()
            
            
            
            
            
            
            
            






    
    
    