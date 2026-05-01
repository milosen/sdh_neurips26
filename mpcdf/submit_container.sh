#!/bin/bash
sbatch <<EOT
#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./mpcdf/slurm_logs/b_array.out.%j
#SBATCH -e ./mpcdf/slurm_logs/b_array.err.%j
# Initial working directory:
#SBATCH -D ./
# Job name
#SBATCH -J apptainer
#
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=64GB
#
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#
#SBATCH --mail-type=none
#SBATCH --mail-user=nmilosevic@cbs.mpg.de
#SBATCH --time=24:00:00

source /etc/profile.d/modules.sh
module purge
module load apptainer
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# For pinning threads correctly:
export OMP_PLACES=cores

srun apptainer exec --nv --containall --bind /usr/share/glvnd,.:"$HOME" ../task_geom/container/apptainer $@

exit 0
EOT
