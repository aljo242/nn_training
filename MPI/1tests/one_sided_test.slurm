#!/bin/bash
#SBATCH --job-name=one_sided_test
#SBATCH --nodes=4 #number of nodes requested
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --cluster=mpi # mpi, gpu and smp are available in H2P
#SBATCH --output=one_sided_test_gather.out
#SBATCH --mem=4g
#SBATCH --mail-user=amj92@pitt.edu #send email to this address if ...
#SBATCH --mail-type=END,FAIL # ... job ends or fails
#SBATCH --time=0-00:01:00 # 6 hours walltime in dd-hh:mm format
#SBATCH --qos=short # required if walltime is greater than 3 days

module purge #make sure the modules environment is sane
module load gcc/8.2.0
module load openmpi/3.1.1
source ~/.bashrc


cd ~/nn_training/MPI/tests
mpirun -n 4 ./one_sided_test --mca mpi_cuda_support 0

