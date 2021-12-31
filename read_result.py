#!/usr/bin/env python3
import argparse
import numpy as np

def drop_flags(label):
    new_label = ""
    for c in label:
        if c in ["i", "s"]:
            pass
        else:
            new_label = new_label + c
    return new_label

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rankings', type=str, required=True, help="tsv file with ranking results")
    parser.add_argument('--pool-size', type=int, required=True, help="number of examples used for ranking after dedup")
    args = parser.parse_args()

    # read the ranking results
    with open(args.rankings, "rt") as f:
        data = f.readlines()
    data = [d.strip().split("\t") for d in data]

    # label rank(index 0) s1 s2
    labels = [d[0] for d in data]
    ranks = [int(d[1]) for d in data]

    # disregard all `i` and `s` flags
    labels = [drop_flags(l) for l in labels]

    # we want the rankings for 1, 2, 3, 4a, and 4, and top1 accuracy for all paraphrases
    assert len(labels)==len(ranks)

    # top1 acc for all paraphrases
    no_all_paraphrases = 0
    no_top1 = 0
    for i in range(len(labels)):
        if "3" in labels[i] or "4" in labels[i]:
            no_all_paraphrases += 1
            if ranks[i]==0:
                no_top1 += 1
    #print("TOP1 (all paraphrases)", no_top1, "/", no_all_paraphrases, str(np.round(no_top1/no_all_paraphrases, 4)),
    #      sep="\t")

    # rankings
    ranks_1 = []
    ranks_2 = []
    ranks_3 = []
    ranks_4a = []
    ranks_4 = []
    for i in range(len(labels)):
        if labels[i]=="1":
            ranks_1.append(ranks[i]+1)
        elif labels[i]=="2":
            ranks_2.append(ranks[i]+1)
        elif labels[i]=="3":
            ranks_3.append(ranks[i]+1)
        elif labels[i]=="4a":
            ranks_4a.append(ranks[i]+1)
        elif labels[i]=="4":
            ranks_4.append(ranks[i]+1)
    print("RESULTS",
          str(np.round(np.mean(ranks_1)/args.pool_size, 4)),
          str(np.round(np.mean(ranks_2)/args.pool_size, 4)),
          str(np.round(np.mean(ranks_3)/args.pool_size, 4)),
          str(np.round(np.mean(ranks_4a)/args.pool_size, 4)),
          str(np.round(np.mean(ranks_4)/args.pool_size, 4)),
          str(np.round(no_top1/no_all_paraphrases, 4)),
          sep="\t")
