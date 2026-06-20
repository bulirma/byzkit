#!/bin/sh
#SBATCH --time=12:00:00
#SBATCH --job-name=byz-ctc
#SBATCH --partition=gpu-ffa-gpulab
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --gres=gpu:V100:1,vram:16G
#SBATCH --output=logs/train_cyril.out.txt
#SBATCH --error=logs/train_cyril.err.txt

DATADIR="___currentdir___"
WORKDIR="/scratch/tmp/___username___"

mkdir "$WORKDIR"

cp -r "$DATADIR/sds12h.lmdb" "$WORKDIR"
cp -r "$DATADIR/common" "$WORKDIR"
cp -r "$DATADIR/train" "$WORKDIR"

cd "$WORKDIR"

. "$DATADIR/venv/bin/activate"
python train/main.py --dataset sds12h.lmdb --model cyril --batch_size 64 --epochs 20 --max_image_height 160
deactivate

cp -r "$WORKDIR/cyril" "$DATADIR"

rm -rf "$WORKDIR"
