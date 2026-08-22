#!/bin/bash
#
#SBATCH --job-name sinteractive
#SBATCH --partition main
#SBATCH --qos normal

#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --nodes=1
#SBATCH -o /home/mazya/OTA_RIS/interactive_%J.out

#export NODE_TLS_REJECT_UNAUTHORIZED='0'

echo Wed 05 Aug 2026 13:24:58 IDT
launch='sleep 600'
echo " STARTING JOB WITH THE COMMAND:  $launch "
while true; do
        eval $launch
done
