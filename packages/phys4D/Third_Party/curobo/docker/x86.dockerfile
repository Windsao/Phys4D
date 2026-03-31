









FROM nvcr.io/nvidia/pytorch:23.08-py3 AS torch_cuda_base

LABEL maintainer "User Name"




RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


RUN apt-get update && apt-get install -y --no-install-recommends \
        pkg-config \
        libglvnd-dev \
        libgl1-mesa-dev \
        libegl1-mesa-dev \
        libgles2-mesa-dev && \
    rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute


RUN apt-get update && apt-get install -y \
  tzdata \
  software-properties-common \
  && rm -rf /var/lib/apt/lists/* \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
  && echo "America/Los_Angeles" > /etc/timezone \
  && dpkg-reconfigure -f noninteractive tzdata \
  && add-apt-repository -y ppa:git-core/ppa \
  && apt-get update && apt-get install -y \
  curl \
  lsb-core \
  wget \
  build-essential \
  cmake \
  git \
  git-lfs \
  iputils-ping \
  make \
  openssh-server \
  openssh-client \
  libeigen3-dev \
  libssl-dev \
  python3-pip \
  python3-ipdb \
  python3-tk \
  python3-wstool \
  sudo git bash unattended-upgrades \
  apt-utils \
  terminator \
  glmark2 \
  && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install --reinstall -y \
  libmpich-dev \
  hwloc-nox libmpich12 mpich \
  && rm -rf /var/lib/apt/lists/*


ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"



ENV TORCH_CUDA_ARCH_LIST "7.0+PTX"
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"


ARG CACHE_DATE=2024-07-19


RUN pip install "robometrics[evaluator] @ git+https://github.com/fishbotics/robometrics.git"







RUN mkdir /pkgs && cd /pkgs && git clone https:

RUN cd /pkgs/curobo && pip3 install .[dev,usd] --no-build-isolation

WORKDIR /pkgs/curobo





ENV PYOPENGL_PLATFORM=egl



RUN echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' >> /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN apt-get update && \
    apt-get install -y libgoogle-glog-dev libgtest-dev curl libsqlite3-dev libbenchmark-dev && \
    cd /usr/src/googletest && cmake . && cmake --build . --target install && \
    rm -rf /var/lib/apt/lists/*

RUN cd /pkgs &&  git clone https:
    cd nvblox && cd nvblox && mkdir build && cd build && \
    cmake .. -DPRE_CXX11_ABI_LINKABLE=ON && \
    make -j32 && \
    make install

RUN cd /pkgs && git clone https:
    cd nvblox_torch && \
    sh install.sh $(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)') && \
    python3 -m pip install -e .

RUN python -m pip install pyrealsense2 opencv-python transforms3d


RUN python -m pip install "robometrics[evaluator] @ git+https://github.com/fishbotics/robometrics.git"


RUN export LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH