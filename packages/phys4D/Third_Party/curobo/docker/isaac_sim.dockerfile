









ARG DEBIAN_FRONTEND=noninteractive
ARG BASE_DIST=ubuntu20.04
ARG CUDA_VERSION=11.4.2
ARG ISAAC_SIM_VERSION=4.0.0


FROM nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION} AS isaac-sim

FROM nvcr.io/nvidia/cudagl:${CUDA_VERSION}-devel-${BASE_DIST}





LABEL maintainer "User Name"

ARG VULKAN_SDK_VERSION=1.3.224.1





RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections












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
  && rm -rf /var/lib/apt/lists/*




RUN apt-get update && apt-get install -y --no-install-recommends \
    libatomic1 \
    libegl1 \
    libglu1-mesa \
    libgomp1 \
    libsm6 \
    libxi6 \
    libxrandr2 \
    libxt6 \
    libfreetype-dev \
    libfontconfig1 \
    openssl \
    libssl1.1 \
    wget \
    vulkan-utils \
&& apt-get -y autoremove \
&& apt-get clean autoclean \
&& rm -rf /var/lib/apt/lists/*




RUN wget -q --show-progress \
    --progress=bar:force:noscroll \
    https:
    -O /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz \
    && echo "Installing Vulkan SDK ${VULKAN_SDK_VERSION}" \
    && mkdir -p /opt/vulkan \
    && tar -xf /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz -C /opt/vulkan \
    && mkdir -p /usr/local/include/ && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/include/* /usr/local/include/ \
    && mkdir -p /usr/local/lib && cp -ra /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/* /usr/local/lib/ \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/lib/libVkLayer_*.so /usr/local/lib \
    && mkdir -p /usr/local/share/vulkan/explicit_layer.d \
    && cp /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/etc/vulkan/explicit_layer.d/VkLayer_*.json /usr/local/share/vulkan/explicit_layer.d \
    && mkdir -p /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/share/vulkan/registry/* /usr/local/share/vulkan/registry \
    && cp -a /opt/vulkan/${VULKAN_SDK_VERSION}/x86_64/bin/* /usr/local/bin \
    && ldconfig \
    && rm /tmp/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.gz && rm -rf /opt/vulkan



ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all


EXPOSE 47995-48012/udp \
       47995-48012/tcp \
       49000-49007/udp \
       49000-49007/tcp \
       49100/tcp \
       8011/tcp \
       8012/tcp \
       8211/tcp \
       8899/tcp \
       8891/tcp

ENV OMNI_SERVER http:



ENV MIN_DRIVER_VERSION 525.60.11


COPY --from=isaac-sim /isaac-sim /isaac-sim
RUN mkdir -p /root/.nvidia-omniverse/config
COPY --from=isaac-sim /root/.nvidia-omniverse/config /root/.nvidia-omniverse/config
COPY --from=isaac-sim /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json
COPY --from=isaac-sim /etc/vulkan/icd.d/nvidia_icd.json /etc/vulkan/implicit_layer.d/nvidia_layers.json

WORKDIR /isaac-sim


ENV TORCH_CUDA_ARCH_LIST="7.0+PTX"





ENV omni_python='/isaac-sim/python.sh'

RUN echo "alias omni_python='/isaac-sim/python.sh'" >> ~/.bashrc



ARG CACHE_DATE=2024-04-11








RUN mkdir /pkgs && cd /pkgs && git clone https:

RUN $omni_python -m pip install ninja wheel tomli


RUN cd /pkgs/curobo && $omni_python -m pip install .[dev] --no-build-isolation

WORKDIR /pkgs/curobo



RUN apt-get update && \
    apt-get install -y curl tcl && \
    rm -rf /var/lib/apt/lists/*






RUN cd /pkgs && wget https:
    tar -xvzf cmake-3.27.1.tar.gz && \
    apt update &&  apt install -y build-essential checkinstall zlib1g-dev libssl-dev && \
    cd cmake-3.27.1 && ./bootstrap && \
    make -j8 && \
    make install &&  rm -rf /var/lib/apt/lists/*


ENV USE_CX11_ABI=0
ENV PRE_CX11_ABI=ON



RUN cd /pkgs && git clone https:
    cd /pkgs/sqlite && CFLAGS=-fPIC ./configure --prefix=/pkgs/sqlite/install/ && \
    make && make install



RUN cd /pkgs && git clone https:
    cd glog && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=/pkgs/glog/install/ \
    -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    && make -j8 && make install


RUN cd /pkgs && git clone https:
    cd gflags &&  \
    mkdir build && cd build && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_INSTALL_PREFIX=/pkgs/gflags/install/ \
    -DGFLAGS_BUILD_STATIC_LIBS=ON -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    && make -j8 && make install

RUN cd /pkgs &&  git clone https:
    mkdir build && cd build && \
    cmake ..  -DBUILD_REDISTRIBUTABLE=ON \
    -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI}  -DPRE_CXX11_ABI_LINKABLE=${PRE_CX11_ABI} \
    -DSQLITE3_BASE_PATH="/pkgs/sqlite/install/" -DGLOG_BASE_PATH="/pkgs/glog/install/" \
    -DGFLAGS_BASE_PATH="/pkgs/gflags/install/" -DCMAKE_CUDA_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    -DBUILD_TESTING=OFF && \
    make -j32 && \
    make install


RUN cd /pkgs/glog && \
    mkdir build_isaac && cd build_isaac && \
    cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF -DBUILD_SHARED_LIBS=OFF -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=${USE_CX11_ABI} \
    && make -j8 && make install

RUN cd /pkgs && git clone https:
    cd /pkgs/nvblox_torch && \
    sh install_isaac_sim.sh $($omni_python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)') && \
    $omni_python -m pip install -e .


RUN $omni_python -m pip install pyrealsense2 opencv-python transforms3d

RUN $omni_python -m pip install "robometrics[evaluator] @ git+https://github.com/fishbotics/robometrics.git"
