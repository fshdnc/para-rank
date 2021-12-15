#!/bin/bash
#SBATCH --job-name=rank
#SBATCH --account=project_2002820
#SBATCH --time=00:15:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=small
#--gres=gpu:v100:1
#--gres=nvme:10
#SBATCH -e /scratch/project_2002820/lihsin/para-rank/results/journal-ppr-%j.err
#SBATCH -o /scratch/project_2002820/lihsin/para-rank/results/journal-ppr-%j.out

min="$1"
max="$2"
analyzer="$3"

set -euo pipefail
echo "START: $(date)"

module purge
module load python-data/3.7.6-1

#--data ../Turku-paraphrase-corpus/agg_data/test.tsv
# ../Turku-paraphrase-corpus/testset/opus-pb-test.tsv
python3 baseline.py --min $min \
    --max $max \
    --analyzer $analyzer \
    --prt \
    --data /scratch/project_2002820/lihsin/datasets/Turku-paraphrase-corpus/data-fi/test.tsv

seff $SLURM_JOBID
echo "END: $(date)"
