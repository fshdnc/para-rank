#!/usr/bin/env python3

from sklearn.metrics.pairwise import cosine_similarity

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
    txt2 = [d[2].strip() for d in data]
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
    txt2 = [d[4].strip() for d in data]
    return labels, txt1, txt2

def label_scheme(labels):
    # remove s
    # subsumption -> a
    new_labels = [l.replace("<","a").replace("s","").replace(">","a") for l in labels]
    return new_labels

def rank(vector_no_order, vector_order, correct_order):
    # takes in two vectors, return the ranking
    ranks = []
    sim_matrix = cosine_similarity(vector_no_order, vector_order)
    for index, sims in enumerate(sim_matrix):
        sims = [(i,sim) for i,sim in enumerate(sims)]
        sims.sort(key=lambda x:x[1], reverse=True)
        rank = [i for i, sim in sims][1:]
        rank = rank.index(correct_order[index])
        ranks.append(rank)
    return ranks
