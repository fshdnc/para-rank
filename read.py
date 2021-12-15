#!/usr/bin/env python3

from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.neighbors import DistanceMetric

#dist = DistanceMetric.get_metric('canberra')

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

def rank(vector_order, vector_no_order, correct_indices, mapping):
    # takes in two vectors, return the ranking
    ranks = []
    sim_matrix = cosine_similarity(vector_order, vector_no_order) #; print("Using cosine similarity")
    #sim_matrix = dist.pairwise(vector_order, vector_no_order); print("Using Canberra distance")
    assert len(sim_matrix)==len(correct_indices)
    for index, sims in enumerate(sim_matrix):
        sims = [(i,sim) for i,sim in enumerate(sims)]
        sims.sort(key=lambda x:x[1], reverse=True)
        rank = [i for i, sim in sims if i not in mapping[index]] #mapping {i:(self_indices)}
        assert len(rank)<len(correct_indices) # the same one has to be removed
        r = []
        for i in mapping[correct_indices[index]]:
            r.append(rank.index(i))
        rank = min(r)
        ranks.append(rank)
    return ranks
