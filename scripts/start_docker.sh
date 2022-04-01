cd $(dirname "$0")
cd ../
docker run --gpus all --rm -it -v ${PWD}:/deep_navier_stokes --ipc=host nvidia-pytorch:latest
