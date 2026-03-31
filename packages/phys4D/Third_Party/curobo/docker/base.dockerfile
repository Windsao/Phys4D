










FROM nvcr.io/nvidia/pytorch:23.08-py3 AS torch_cuda_base

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && apt-get install -y \
  tzdata \
  && rm -rf /var/lib/apt/lists/* \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
  && echo "America/Los_Angeles" > /etc/timezone \
  && dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update &&\
    apt-get install -y sudo git bash software-properties-common graphviz &&\
    rm -rf /var/lib/apt/lists/*



RUN python -m pip install --upgrade pip && python3 -m pip install graphviz
