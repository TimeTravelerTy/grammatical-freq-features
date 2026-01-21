#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=04:00:00
#$ -N probe_features_job
#$ -o ./logs/$JOB_NAME.$JOB_ID.out
#$ -e ./logs/$JOB_NAME.$JOB_ID.err
#$ -V

set -e

module purge
module load cuda
# --- activate miniconda ---
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate py311

python probe_features.py   --model_name meta-llama/Meta-Llama-3-8B   --ud_train_file data/en_ewt-ud/en_ewt-ud-train.conllu   --concept_key Number   --concept_value Sing
python probe_features.py   --model_name meta-llama/Meta-Llama-3-8B   --ud_train_file data/en_ewt-ud/en_ewt-ud-train.conllu   --concept_key Number   --concept_value Plur
python probe_features.py   --model_name meta-llama/Meta-Llama-3-8B   --ud_train_file data/en_ewt-ud/en_ewt-ud-train.conllu   --concept_key Tense   --concept_value Pres
python probe_features.py   --model_name meta-llama/Meta-Llama-3-8B   --ud_train_file data/en_ewt-ud/en_ewt-ud-train.conllu   --concept_key Tense   --concept_value Past
