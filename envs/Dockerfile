FROM nvidia/cuda:12.9.1-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=US

RUN apt-get -y update -qq && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        libgomp1 \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
        wget \
        zip \
    && \
    rm -rf /var/lib/apt/lists/*
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib64"

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip setuptools wheel && \
    pip install torch \
                gymnasium \
                ale-py \
                cloudpickle \
                matplotlib \
                opencv-python-headless \
                psutil \
                pytz \
                tensorboard \
                tqdm

RUN git clone -b master --recursive https://github.com/NVLabs/cule && \
    cd cule && \
    git -C third_party/pybind11 fetch --tags && \
    git -C third_party/pybind11 checkout v2.13.6 && \
    pip install --no-build-isolation .
