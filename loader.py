import torch
import numpy as np 
from torch.utils.data import Sampler
import kaldiio
from utils import pad_list
from torch.utils.data import DataLoader
import configargparse

    
def create_loader(
    data: dict ,
    params: configargparse.Namespace,
    pin_memory: bool=False,
    min_batch_size: int=1 ,
    shortest_first: bool=False):
    """
    Creates batches with different batch sizes which maximizes
    the number of bins up to `batch_bins`.
    :param Dict[str, Dict[str, Any]] sorted_data: dictionary loaded from data.json
    :param int batch_bins: Maximum frames of a batch
    :param int min_batch_size: minimum batch size (for multi-gpu)
    :param bool shortest_first: Sort from batch with shortest samples
        to longest if true, otherwise reverse
    :returns: List[Tuple[str, Dict[str, List[Dict[str, Any]]]] list of batches
    """
    batch_bins = params.batch_bins
    print("Batch Bins : {}".format(batch_bins))
    sorted_data = sorted(
            data.items(),
            key=lambda data: int(data[1]["input"][0]["shape"][0]),
            reverse=not shortest_first,
        )
    if batch_bins <= 0:
        raise ValueError(f"invalid batch_bins={batch_bins}")
    length = len(sorted_data)
    idim = int(sorted_data[0][1]["input"][0]["shape"][1])
    odim = int(sorted_data[0][1]["output"][0]["shape"][1]) if "output" in sorted_data[0][1].keys() else 0
    minibatches = []
    start = 0
    n = 0
    while True:
        # Dynamic batch size depending on size of samples
        b = 0
        next_size = 0
        max_olen = 0
        while next_size < batch_bins and (start + b) < length:
            ilen = int(sorted_data[start + b][1]["input"][0]["shape"][0]) * idim
            olen = int(sorted_data[start + b][1]["output"][0]["shape"][0]) * odim if "output" in sorted_data[0][1].keys() else 0
            if olen > max_olen:
                max_olen = olen
            next_size = (max_olen + ilen) * (b + 1)
            if next_size <= batch_bins:
                b += 1
            elif next_size == 0:
                raise ValueError(
                    f"Can't fit one sample in batch_bins ({batch_bins}): "
                    f"Please increase the value"
                )
        end = min(length, start + max(min_batch_size, b))
        batch = [element[0] for element in sorted_data[start:end]]
        if shortest_first:
            batch.reverse()
        minibatches.append(batch)
        # Check for min_batch_size and fixes the batches if needed
        i = -1
        while len(minibatches[i]) < min_batch_size:
            missing = min_batch_size - len(minibatches[i])
            if -i == len(minibatches):
                minibatches[i + 1].extend(minibatches[i])
                minibatches = minibatches[1:]
                break
            else:
                minibatches[i].extend(minibatches[i - 1][:missing])
                minibatches[i - 1] = minibatches[i - 1][missing:]
                i -= 1
        if end == length:
            break
        start = end
        n += 1
    lengths = [len(x) for x in minibatches]
    print(" #Utts: {} | Created {} minibatches containing {} to {} samples, and on average {} samples".format(len(sorted_data),len(minibatches),min(lengths),max(lengths),int(np.mean(lengths)) ) )
    
    dataset = ASRDataset(data,params)
    loader = DataLoader(dataset=dataset,batch_sampler=minibatches,num_workers=params.nworkers,collate_fn=dataset.collate_function)

    return dataset,loader,minibatches
              
    
        
class ASRDataset(torch.utils.data.Dataset):
    def __init__(self,data:dict,params: configargparse.Namespace):
        """
        :param dict data- The json data file dictionary
        :param Namespace params- Training options
        """
        self.data = data
        self.audio_pad = params.audio_pad 
        self.text_pad = params.text_pad
    
    def __len__(self):
        """
        Returns the number of examples in the dataset
        """
        return len(self.data.keys())

    def __getitem__(self,idx: str):
        """
        Retrieves an item from the dataset given the index
        :param str idx- Utterance ID key
        :returns np.array audio_feat- Speech features
        :returns list feat_len- Speech sequence length
        :returns np.array target- Output sequence 
        :returns list target_len- Token sequence length
        :returns str idx- Utterance ID key
        """
        audio_feat = kaldiio.load_mat(self.data[idx]["input"][0]["feat"])
        feat_len = self.data[idx]["input"][0]["shape"][0]
        target = np.array([int(x) for x in self.data[idx]["output"][0]["tokenid"].split(' ')]) if "output" in self.data[idx].keys() else None
        target_len = len(target) if target is not None else 0
        return (audio_feat,feat_len,target,target_len,idx)
        
    def collate_function(self,batch):
        """
        Retrieves an item from the dataset given the index
        :param generator batch- Batch of data
        :returns np.array padded_feats- Speech features
        :returns list feat_lens- Speech sequence length
        :returns np.array padded_targets- Output sequence 
        :returns list target_lens- Token sequence length
        :returns list utt_keys- Utterance ID key
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        padded_feats = pad_list([torch.from_numpy(x[0]).float() for x in batch],self.audio_pad).to(device)
        padded_targets = pad_list([torch.from_numpy(x[2]).long() for x in batch],self.text_pad).to(device) if batch[0][2] is not None else None
        feat_lens = [x[1] for x in batch]
        target_lens = [x[3] for x in batch] if batch[0][2] is not None else None
        utt_keys = [x[4] for x in batch]
        return padded_feats,feat_lens,padded_targets,target_lens,utt_keys
