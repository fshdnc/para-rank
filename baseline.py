#!/usr/bin/env python3

# tf-idf baseline

import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from read import *

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
    parser.add_argument('--min', type=int, default=2, help="TFIDF minimum")
    parser.add_argument('--max', type=int, default=5, help="TFIDF maximum")
    parser.add_argument('--analyzer', type=str, default="char_wb", help="TFIDF analyzer, word, char, or char_wb")
    args = parser.parse_args()

    # read data - need s1, s2, label
    labels, txt1, txt2 = [], [], []
    for data in args.data:
        labels_s, txt1_s, txt2_s = read_tsv(data)
        labels.extend(labels_s); txt1.extend(txt1_s); txt2.extend(txt2_s)
    txt1 = [t.lower() for t in txt1]
    txt2 = [t.lower() for t in txt2]

    txt = list(set(txt1 + txt2))
    # mapping
    mapping = {i:[j for j,t in enumerate(txt) if s==t] for i,s in enumerate(txt1+txt2)}

    # indices of the correct answer
    correct_indices = [i+len(txt1) for i in range(len(txt2))] + [i for i in range(len(txt1))]
    
    # tf-idf vectorization
    vectorizer = TfidfVectorizer(ngram_range=(args.min, args.max), analyzer=args.analyzer) #, stop_words=stop_words)
    vectorizer.fit(txt1+txt2)
    txt_encoded = vectorizer.transform(txt)
    txt_order_encoded = vectorizer.transform(txt1+txt2)
    
    # rank
    ranks = rank(txt_order_encoded, txt_encoded, correct_indices, mapping)
    labels = labels + labels
    assert len(ranks)==len(labels)
    
    # results
    results = ["RESULTS"]
    for label in ["1","2","3","4ai","4a","4i","4"]:
        label_rank = [r for l, r in zip(labels, ranks) if l==label]
        #print("Label\t{}\tOcc\t{}\tAvg_rank\t{:.3f}".format(label, len(label_rank),np.mean(label_rank)))
        results.append(str(np.round(np.mean(label_rank),3)))
        if label=="4":
            top1 = sum([1 for r in label_rank if r==0])
            print("TOP_1_ACC\t{:.4f} ({}/{})".format(top1/len(label_rank), top1, len(label_rank)))
    print("TF-IDF baseline")
    print("\t".join(results))
    


