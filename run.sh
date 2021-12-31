#!/bin/bash
#SBATCH --job-name=rank
#SBATCH --account=project_2002820
#SBATCH --time=00:60:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#--gres=nvme:10 #non-gpu partition
#SBATCH -e /scratch/project_2002820/lihsin/para-rank/results/journal-ppr-%j.err
#SBATCH -o /scratch/project_2002820/lihsin/para-rank/results/journal-ppr-%j.out

set -euo pipefail
echo "START: $(date)"

number="$1"
sbert_model=/scratch/project_2002820/lihsin/sentence-transformers/finnish-SBERT/output/${number}

module purge
module load pytorch/1.1.0
source /projappl/project_2002820/venv/SBERT/bin/activate

# version 2.1.0
#module load pytorch/1.9
# git clone the repo first
#export PYTHONPATH=/scratch/project_2002820/lihsin/para-rank/sentence-transformers

#export TRANSFORMERS_CACHE=$LOCAL_SCRATCH #non-gpu partition

echo "Model: $sbert_model"
echo "Turku paraphrase test"
python3 rank.py \
    --prt \
    --data /scratch/project_2002820/lihsin/datasets/Turku-paraphrase-corpus/data-fi/test.tsv \
    --sbert $sbert_model

echo "opus-parsebank test"
python3 rank.py \
    --prt \
    --data /scratch/project_2002820/lihsin/Turku-paraphrase-corpus/testset/opus-pb-test.tsv \
    --sbert $sbert_model

# --data ../Turku-paraphrase-corpus/agg_data/test.tsv \ # old expt in June 2021
seff $SLURM_JOBID
echo "END: $(date)"
