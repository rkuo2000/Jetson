export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH

# python3 -m pip install pip
# python3 -m venv .yolo
source ~/.yolo/bin/activate  # commented out by conda initialize
cd ~
