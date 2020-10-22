
sudo apt-get update
sudo apt-get install -y \
    apt-transport-https \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    git \
    libatlas-base-dev \
    libcurl4-openssl-dev \
    libjemalloc-dev \
    liblapack-dev \
    libopenblas-dev \
    libopencv-dev \
    libzmq3-dev \
    ninja-build \
    python-dev \
    software-properties-common \
    sudo \
    unzip \
    virtualenv \
    wget
sudo apt-get install -y libgfortran3

wget https://mxnet-public.s3.amazonaws.com/install/raspbian/mxnet-1.5.0-py2.py3-none-any.whl
pip install mxnet-1.5.0-py2.py3-none-any.whl
