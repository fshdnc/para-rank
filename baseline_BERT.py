#!/usr/bin/env python3

# bert baseline
import torch
import argparse
import numpy as np
import transformers

from read import *

bert_model = transformers.BertModel.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")
bert_model.eval()
if torch.cuda.is_available():
    bert_model = bert_model.cuda()
bert_tokenizer = transformers.BertTokenizer.from_pretrained("TurkuNLP/bert-base-finnish-cased-v1")

def embed(data,bert_model,how_to_pool="CLS"):
    with torch.no_grad(): #tell the model not to gather gradients
        mask=data.clone().float() #
        mask[data>0]=1.0
        emb=bert_model(data.cuda(),attention_mask=mask.cuda()) #runs BERT and returns several things, we care about the first
        #emb[0]  # batch x word x embedding
        if how_to_pool=="AVG":
            pooled=emb[0]*(mask.unsqueeze(-1)) #multiply everything by the mask
            pooled=pooled.sum(1)/mask.sum(-1).unsqueeze(-1) #sum and divide by non-zero elements in mask to get masked average
        elif how_to_pool=="CLS":
            pooled=emb[0][:,0,:].squeeze() #Pick the first token as the embedding
        else:
            assert False, "how_to_pool should be CLS or AVG"
            print("Pooled shape:",pooled.shape)
    return pooled.cpu().numpy() #done! move data back to CPU and extract the numpy array

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
    txt1_encoded = embed(txt1,bert_model,how_to_pool="CLS")
    txt2_encoded = embed(txt2,bert_model,how_to_pool="CLS")
    
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
    


