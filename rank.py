#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from read import *

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

def print_ranking(vec1, vec2, txt1, txt2, labels):
    import sys
    # prints the sentences, labels,
    # and ranking to stderr
    sim_matrix = cosine_similarity(vec1, vec2)
    print("Label\trank\ts1\ts2\t", file=sys.stderr)
    for index, sims in enumerate(sim_matrix):
        sims = [(i,sim) for i,sim in enumerate(sims)]
        sims.sort(key=lambda x:x[1], reverse=True)
        rank = [i for i, sim in sims]
        rank = rank.index(index)
        print(labels[index], rank, txt1[index], txt2[index], sep="\t", file=sys.stderr)
    
if __name__=="__main__":
    """
    for sentpair in data:
        get s2 ranking
    for label in 4, 4sub, 3, 2, 1:
        average ranking
    output number
    """
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+', required=True, help="tsv file")
    parser.add_argument('--sbert', required=True)
    parser.add_argument("--prt", type=str2bool, nargs='?', const=True, default=False, help="Print ranking results to stderr.")
    parser.add_argument("--rank", type=str2bool, nargs='?', const=False, default=True, help="Print overall ranking statistics.")
    args = parser.parse_args()

    # read data - need s1, s2, label
    labels, txt1, txt2 = [], [], []
    for data in args.data:
        labels_s, txt1_s, txt2_s = read_tsv(data)
        labels.extend(labels_s); txt1.extend(txt1_s); txt2.extend(txt2_s)
        
    # encode by sbert
    model = SentenceTransformer(args.sbert)
    txt1_encoded = model.encode(txt1)
    txt2_encoded = model.encode(txt2)

    if args.rank:
        # rank
        ranks = rank(txt1_encoded, txt2_encoded)
        assert len(ranks)==len(labels)
    
        # results
        results = []
        for label in ["1","2","3","4ai","4a","4i","4"]:
            label_rank = [r for l, r in zip(labels, ranks) if l==label]
            #print("Label\t{}\tOcc\t{}\tAvg_rank\t{:.3f}".format(label, len(label_rank),np.mean(label_rank)))
            results.append(str(np.round(np.mean(label_rank),3)))
        print("SBERT model:",args.sbert)
        print("\t".join(results))

    if args.prt:
        print_ranking(txt1_encoded, txt2_encoded, txt1, txt2, labels)


