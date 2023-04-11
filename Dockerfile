FROM nvcr.io/nvidia/pytorch:22.01-py3

RUN apt-get install curl build-essential -y
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
# RUN apt-get update -y
# RUN apt-get install -y kmod
# RUN wget https://us.download.nvidia.com/tesla/510.108.03/NVIDIA-Linux-x86_64-510.108.03.run
# RUN sh NVIDIA-Linux-x86_64-510.108.03.run
RUN apt-get update \
&& pip install --upgrade pip setuptools \
&& pip install --upgrade pip

# Clear cache
RUN apt-get clean && rm -rf /var/lib/apt/lists/*1
