#!/usr/bin/env python3

# bert baseline
import torch
import argparse
import numpy as np
from math import ceil
import transformers

from read import *

bert_model = transformers.BertModel.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
bert_model.eval()
if torch.cuda.is_available():
    bert_model = bert_model.cuda()
bert_tokenizer = transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")

def tokenize(text):
    tokenized_ids=[bert_tokenizer.encode(txt,add_special_tokens=True) for txt in text] #this runs the BERT tokenizer, returns list of lists of integers
    tokenized_ids_t=[torch.tensor(ids,dtype=torch.long) for ids in tokenized_ids] #turn lists of integers into torch tensors
    tokenized_single_batch=torch.nn.utils.rnn.pad_sequence(tokenized_ids_t,batch_first=True)
    #bert_embedded=embed(tokenized_single_batch,bert_model)
    #if len(self.lines_and_tokens)==1:
    #    self.bert_embedded=self.bert_embedded.reshape(1, -1)
    return tokenized_single_batch

def embed(data_whole,bert_model,how_to_pool="CLS", batch_size=128):
    embedded = None
    for i in range(ceil(len(data_whole/batch_size))):
        # the embedding by BERT process
        with torch.no_grad(): #tell the model not to gather gradients
            data = data_whole[i*batch_size: (i+1)*batch_size]
            if len(data)==0:
                break
            mask=data.clone().float() #
            mask[data>0]=1.0
            mask = mask.cuda()
            emb=bert_model(data.cuda(),attention_mask=mask.cuda()) #runs BERT and returns several things, we care about the first
            #emb[0]  # batch x word x embedding
            if how_to_pool=="AVG":
                pooled=emb[0]*(mask.unsqueeze(-1)) #multiply everything by the mask
                pooled=pooled.sum(1)/mask.sum(-1).unsqueeze(-1) #sum and divide by non-zero elements in mask to get masked average
            elif how_to_pool=="MAX":
                pooled=emb[0]*(mask.unsqueeze(-1)) #multiply everything by the mask
                pooled = torch.max(pooled, 1)[0]
            elif how_to_pool=="CLS":
                pooled=emb[0][:,0,:].squeeze() #Pick the first token as the embedding
            else:
                assert False, "how_to_pool should be CLS or AVG"
                print("Pooled shape:",pooled.shape)
            pooled = pooled.cpu().numpy() #done! move data back to CPU and extract the numpy array
        if not isinstance(embedded, np.ndarray):
            embedded = pooled
        else:
            embedded = np.concatenate((embedded, pooled), axis=0)
    return embedded

if __name__=="__main__":
    """
    for sentpair in data:
        get s2 ranking
    for label in 4, 4sub, 3, 2, 1:
        average ranking
    output number
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs="+", required=True)
    args = parser.parse_args()

    # read data - need s1, s2, label
    labels, txt1, txt2 = [], [], []
    for data in args.data:
        labels_s, txt1_s, txt2_s = read_tsv(data)
        labels.extend(labels_s); txt1.extend(txt1_s); txt2.extend(txt2_s)

    # bert embedding
    txt1_encoded = embed(tokenize(txt1), bert_model, how_to_pool="CLS")
    txt2_encoded = embed(tokenize(txt2), bert_model, how_to_pool="CLS")
    
    # rank
    ranks = rank(txt1_encoded, txt2_encoded)
    assert len(ranks)==len(labels)
    
    # results
    results = []
    for label in ["1","2","3","4ai","4a","4i","4"]:
        label_rank = [r for l, r in zip(labels, ranks) if l==label]
        #print("Label\t{}\tOcc\t{}\tAvg_rank\t{:.3f}".format(label, len(label_rank),np.mean(label_rank)))
        results.append(str(np.round(np.mean(label_rank),3)))
    print("BERT baseline")
    print("\t".join(results))
    


