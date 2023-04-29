#!bin/bash
salloc --nodes 1 -n 4 -c 4 --gpus 4 --qos interactive --time 02:40:00 --constraint gpu --account=m4139
#export MPICH_GPU_SUPPORT_ENABLED=1 
#export LD_LIBRARY_PATH=/global/homes/s/shulishu/xanadu/venv/lib64/python3.9/site-packages/cuquantum/lib/:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:$LD_LIBRARY_PATH

#python tests 
#https://docs.nersc.gov/development/languages/python/using-python-perlmutter/#mpi4py-on-perlmutter
module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python gcc
conda activate gpu-aware-mpi
export LD_LIBRARY_PATH=/global/homes/s/shulishu/xanadu/venv/lib64/python3.9/site-packages/cuquantum/lib/:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:$LD_LIBRARY_PATH
module load gcc
export MPICH_GPU_SUPPORT_ENABLED=1 
pip install -e .
srun python -m pytest tests/mpitests

#cpp tests
module load PrgEnv-gnu cray-mpich cudatoolkit craype-accel-nvidia80 python gcc
conda activate gpu-aware-mpi
export LD_LIBRARY_PATH=/global/homes/s/shulishu/xanadu/venv/lib64/python3.9/site-packages/cuquantum/lib/:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:$LD_LIBRARY_PATH
export MPICH_GPU_SUPPORT_ENABLED=1 
#cmake -BBuildTests -DPLLGPU_BUILD_TESTS=On  -DCMAKE_SYSTEM_NAME=CrayLinuxEnvironment -DCUQUANTUM_SDK=../venv/lib64/python3.9/site-packages/cuquantum
cmake -BBuildTests -DPLLGPU_BUILD_TESTS=On 
cmake --build BuildTests
LD_PRELOAD=$PWD/../venv/lib64/python3.9/site-packages/cuquantum/lib/libcustatevec.so.1 srun ./BuildTests/pennylane_lightning_gpu/src/tests/mpi_runner

#pip install -e .
#srun python -m pytest test
#export LD_LIBRARY_PATH=/global/homes/s/shulishu/xanadu/venv/lib64/python3.9/site-packages/cuquantum/lib/:/opt/cray/pe/mpich/8.1.25/ofi/gnu/9.1/lib/:$LD_LIBRARY_PATH