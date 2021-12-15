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

def str2bool(v):
    '''
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    Downside: the 'nargs' might catch a positional argument
    '''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

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
    parser.add_argument('--pooling', type=str, default='AVG', help="AVG, CLS or MAX")
    parser.add_argument("--prt", type=str2bool, nargs='?', const=True, default=False, help="Print ranking results to stderr.")
    args = parser.parse_args()

    # read data - need s1, s2, label
    labels, txt1, txt2 = [], [], []
    for data in args.data:
        labels_s, txt1_s, txt2_s = read_tsv(data)
        labels.extend(labels_s); txt1.extend(txt1_s); txt2.extend(txt2_s)

    txt = list(set(txt1 + txt2))
    # mapping
    mapping = {i:[j for j,t in enumerate(txt) if s==t] for i,s in enumerate(txt1+txt2)}

    # precompute the index of the pair <- bugged because some indices have the same txt
    #txt2_correct_answers = [txt.index(s) for s in txt1]
    #txt1_correct_answers = [txt.index(s) for s in txt2]
    # indices of the correct answer
    correct_indices = [i+len(txt1) for i in range(len(txt2))] + [i for i in range(len(txt1))]
    
    # bert embedding
    txt_encoded = embed(tokenize(txt), bert_model, how_to_pool=args.pooling)
    txt_order_encoded = embed(tokenize(txt1+txt2), bert_model, how_to_pool=args.pooling)
    
    # rank
    ranks = rank(txt_order_encoded, txt_encoded, correct_indices, mapping)
    labels = labels + labels
    assert len(ranks)==len(labels)
    
    # results
    results = ["RESULTS"]
    top1s = ["TOP1"]
    for label in ["1","2","3","4ai","4a","4i","4"]:
        label_rank = [r for l, r in zip(labels, ranks) if l==label]
        if not label_rank: # no example for the label
            results.append("-")
            top1s.append("-")
            continue
        #print("Label\t{}\tOcc\t{}\tAvg_rank\t{:.3f}".format(label, len(label_rank),np.mean(label_rank)))
        results.append(str(np.round(np.mean(label_rank),3)))
        top1 = sum([1 for r in label_rank if r==0])
        top1s.append(str(np.round(top1/len(label_rank),3)))
        #print("TOP_1_ACC\t{:.4f} ({}/{})".format(top1/len(label_rank), top1, len(label_rank)))
    print("BERT baseline:", args)
    print("Number of examples:", len(labels))
    print("\t".join(results))
    print("\t".join(top1s))

    if args.prt:
        import sys
        left = [t for t in txt1+txt2]
        right = [t for t in txt2+txt1]
        for i, r in enumerate(ranks):
            print(labels[i], r, left[i], right[i], sep="\t", file=sys.stderr)
