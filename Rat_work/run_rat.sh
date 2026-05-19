#!/bin/bash
#SBATCH --job-name=liquido-electron
#SBATCH --output=logs/singularity-shell_%a.out
#SBATCH --error=logs/singularity-shell_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclude=p-mc-3471
#SBATCH --partition=basic
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --array=1-10

# Make sure log directory exists
mkdir -p logs

singularity exec /storage/group/dfc13/default/gmwendel/liquido/PSU_Prototype/SimPackage/ratpac-two.sif \
    bash -c "
        source /storage/group/dfc13/default/gmwendel/liquido/PSU_Prototype/SimPackage/ratpac-setup/env.sh && \
        source /storage/group/dfc13/default/gmwendel/liquido/PSU_Prototype/SimPackage/LiquidOSimulations/liquido.sh && \
        liquido /storage/home/wjk5361/work/MPULSESIM/mu_data.mac"