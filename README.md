# Ranking evaluation for sentence embeddings

Use the paraphrase corpus (5000 test set) for ranking
Good embeddings should rank them more or less 4 - 4<> - 3 - 2 - 1.

### Labels
subsumption: `a`
`s` not taken into account

### Evaluate
```
python3 rank.py --data <tsv-path> --sbert <sbert-path>
```

### Baseline
TF-IDF baseline
```
python3 baseline.py --data <tsv-path>
```