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
    txt = list(set(txt1 + txt2))

    # precompute the index of the pair
    txt2_correct_answers = [txt.index(s) for s in txt1]
    txt1_correct_answers = [txt.index(s) for s in txt2]
    
    # tf-idf vectorization
    vectorizer = TfidfVectorizer(ngram_range=(args.min, args.max), analyzer=args.analyzer) #, stop_words=stop_words)
    vectorizer.fit(txt1+txt2)
    txt_encoded = vectorizer.transform(txt)
    txt_order_encoded = vectorizer.transform(txt1+txt2)
    
    # rank
    ranks = rank(txt_order_encoded, txt_encoded, txt1_correct_answers+txt2_correct_answers)
    labels = labels + labels
    assert len(ranks)==len(labels)
    
    # results
    results = []
    for label in ["1","2","3","4ai","4a","4i","4"]:
        label_rank = [r for l, r in zip(labels, ranks) if l==label]
        #print("Label\t{}\tOcc\t{}\tAvg_rank\t{:.3f}".format(label, len(label_rank),np.mean(label_rank)))
        results.append(str(np.round(np.mean(label_rank),3)))
    print("TF-IDF baseline")
    print("\t".join(results))
    


