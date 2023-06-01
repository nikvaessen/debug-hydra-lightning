#!/usr/bin/env bash

#SBATCH --partition=das
#SBATCH --account=das
#SBATCH --qos=das-preempt

#SBATCH --time=4-00:00
#SBATCH --job-name=bash
#SBATCH --output=/home/nvaessen/logs/%A_%a.out
#SBATCH --error=/home/nvaessen/logs/%A_%a.err

#SBATCH --nodes=1
#SBATCH --gres=gpu:rtx_a5000:4

#SBATCH --cpus-per-task=64
#SBATCH --mem=120GB

#SBATCH --mail-user=nvaessen
#SBATCH --mail-type=BEGIN,END,FAIL

# activate venv
source .venv/bin/activate

python main.py devices=4