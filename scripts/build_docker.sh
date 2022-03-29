docker pull nvcr.io/nvidia/pytorch:22.02-py3
cd $(dirname "$0")
docker build --rm -t nvidia-pytorch . -f Dockerfile
