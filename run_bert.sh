#!/bin/bash
#SBATCH --job-name=rank
#SBATCH --account=project_2002820
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#--gres=nvme:10
#SBATCH -e /scratch/project_2002820/lihsin/para-rank/results/journal-ppr-%j.err
#SBATCH -o /scratch/project_2002820/lihsin/para-rank/results/journal-ppr-%j.out

### first argument: poling method, AVG, CLS, or MAX

set -euo pipefail
echo "START: $(date)"

module purge
module load pytorch/1.3.1

#export TRANSFORMERS_CACHE=$LOCAL_SCRATCH
#../Turku-paraphrase-corpus/testset/opus-pb-test.tsv \
#--data ../Turku-paraphrase-corpus/agg_data/test.tsv \
python3 baseline_BERT.py \
    --data /scratch/project_2002820/lihsin/datasets/Turku-paraphrase-corpus/data-fi/test.tsv \
    --prt \
    --pooling "$1"

seff $SLURM_JOBID
echo "END: $(date)"
