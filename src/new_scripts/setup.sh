# Verify CUDA installation
nvcc --version

# Check Python version (3.10+ required)
python3 --version

# Verify GPU accessibility
nvidia-smi

# Check available system memory
free -h

# Docker permission:
docker ps

# if there is permission issue, (e.g., permission denied while trying to connect to the Docker daemon socket), then do:
sudo usermod -aG docker $USER
newgrp docker


docker pull nvcr.io/nvidia/nemo-automodel:26.02
