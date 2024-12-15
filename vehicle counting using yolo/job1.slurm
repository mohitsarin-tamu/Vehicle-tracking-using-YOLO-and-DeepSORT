#!/bin/bash


##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Object-Detection      #Set the job name to "JobExample4"
#SBATCH --time=01:00:00              #Set the wall clock limit to 1hr and 30min
#SBATCH --nodes=8
#SBATCH --mem=17000M                  #Request 2560MB (2.5GB) per node
#SBATCH --output=Object-Dect.%j      #Send stdout/err to "Example4Out.[jobID]"
#SBATCH --gres=gpu:a100:1             #Request 1 "rtx" GPU per node
#SBATCH --partition=gpu              #Request the GPU partition/queue

##OPTIONAL JOB SPECIFICATIONS
##SBATCH --mail-type=ALL              #Send email on all job events
##SBATCH --mail-user=mohitsarin26@tamu.edu    #Send all emails to email_address

source activate final
python3 VehicleCountingYOLO.py
