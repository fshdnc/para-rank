#!/usr/bin/env python3

# tf-idf baseline

import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def read_tsv(filename):
    """
    tsv format (no header)
    label   txt1    txt2
    """
    with open(filename, "r") as f:
        data = f.readlines()
    data = [d.split("\t") for d in data]
    labels = [d[0] for d in data]
    labels = label_scheme(labels)
    txt1 = [d[1] for d in data]
    txt2 = [d[2] for d in data]
    return labels, txt1, txt2

def read_opus_pb_tsv(filename):
    """
    tsv format
    label   source  similarity      txt1    txt2
    """
    with open(filename, "r") as f:
        data = f.readlines()
    data = data[1:] # remove header
    data = [d.split("\t") for d in data]
    labels = [d[0] for d in data]
    labels = label_scheme(labels)
    txt1 = [d[3] for d in data]
    txt2 = [d[4] for d in data]
    return labels, txt1, txt2

def label_scheme(labels):
    # remove s
    # subsumption -> a
    new_labels = [l.replace("<","a").replace("s","").replace(">","a") for l in labels]
    return new_labels

def rank(vec1, vec2):
    # takes in two vectors, return the ranking
    sim_matrix = cosine_similarity(vec1, vec2)
    ranks = []
    for index, sims in enumerate(sim_matrix):
        sims = [(i,sim) for i,sim in enumerate(sims)]
        sims.sort(key=lambda x:x[1], reverse=True)
        rank = [i for i, sim in sims]
        rank = rank.index(index)
        ranks.append(rank)
    return ranks
    
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

    # tf-idf vectorization
    vectorizer = TfidfVectorizer(ngram_range=(args.min, args.max), analyzer=args.analyzer) #, stop_words=stop_words)
    vectorizer.fit(txt1+txt2)
    txt1_encoded = vectorizer.transform(txt1)
    txt2_encoded = vectorizer.transform(txt2)
    
    # rank
    ranks = rank(txt1_encoded, txt2_encoded)
    assert len(ranks)==len(labels)
    
    # results
    results = []
    for label in ["1","2","3","4ai","4a","4i","4"]:
        label_rank = [r for l, r in zip(labels, ranks) if l==label]
        #print("Label\t{}\tOcc\t{}\tAvg_rank\t{:.3f}".format(label, len(label_rank),np.mean(label_rank)))
        results.append(str(np.round(np.mean(label_rank),3)))
    print("TF-IDF baseline")
    print("\t".join(results))
    


