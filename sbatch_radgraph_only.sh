#!/bin/bash

#SBATCH --partition gpu_quad
#SBATCH --mail-type=FAIL,RUNNING,COMPLETE
#SBATCH --mail-user=ktian@college.harvard.edu
#SBATCH --gres=gpu:teslaV100s:1
#SBATCH -c 8
#SBATCH --mem=40G
#SBATCH --time=120:00:00
#SBATCH -o sbatch_m2trans_radgraph_only_%j_run.out
#SBATCH -e sbatch_m2trans_radgraph_only%j_err.out

source ~/.bashrc

module load conda2/4.2.13
module load gcc/6.2.0
module load cuda/11.2

conda activate ifcc
KAT_DIR=/n/data1/hms/dbmi/rajpurkar/lab/home/kt220
cd $KAT_DIR/m2trans-radgraph-only

nvidia-smi

python3 train.py --cuda --corpus mimic-cxr --cache-data cache --epochs 32 --batch-size 24 --rl-epoch 1 --rl-metrics BERTScore,EntityMatchExact --rl-weights 0.01,0.495,0.495 --entity-match mimic-cxr_ner.txt.gz --baseline-model out_m2trans_nll/model_31-152173.dict.gz --img-model densenet --img-pretrained resources/chexpert_auc14.dict.gz --cider-df mimic-cxr_train-df.bin.gz --bert-score distilbert-base-uncased --lr 5e-6 --lr-step 32 /n/data1/hms/dbmi/rajpurkar/lab/datasets/cxr/MIMIC-CXR/ resources/glove_mimic-cxr_train.512.txt.gz out_m2trans_nll-radgraph-only
