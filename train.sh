#!/usr/bin/env bash
#SBATCH -p rtx6000
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --job-name=train_classifier
#SBATCH --qos=normal
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=logs/run_%j.log


echo Running on $(hostname)
date;hostname;pwd

# choose a random port
python -u main.py --lr 1e-6 --num_epochs 10 --num_workers 2 --features best --dataset_size fourty_p --encoder_weights imagenet 