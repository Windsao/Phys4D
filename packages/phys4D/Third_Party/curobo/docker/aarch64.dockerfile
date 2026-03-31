









FROM nvcr.io/nvidia/l4t-pytorch:r35.1.0-pth1.13-py3 AS l4t_pytorch


RUN apt-get update &&\
    apt-get install -y sudo git bash unattended-upgrades glmark2 &&\
    rm -rf /var/lib/apt/lists/*




RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections



RUN apt-get update && apt-get install -y \
  tzdata \
  && rm -rf /var/lib/apt/lists/* \
  && ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
  && echo "America/Los_Angeles" > /etc/timezone \
  && dpkg-reconfigure -f noninteractive tzdata





RUN apt-get update && apt-get install -y \
  curl \
  lsb-core \
  software-properties-common \
  wget \
  && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository -y ppa:git-core/ppa

RUN apt-get update && apt-get install -y \
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

ARG ROS_PKG=ros_base
ENV ROS_DISTRO=noetic
ENV ROS_ROOT=/opt/ros/${ROS_DISTRO}
ENV ROS_PYTHON_VERSION=3

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace





RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          git \
		cmake \
		build-essential \
		curl \
		wget \
		gnupg2 \
		lsb-release \
		ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https:





RUN apt-get update && \
    apt-get install -y --no-install-recommends \
          libpython3-dev \
          python3-rosdep \
          python3-rosinstall-generator \
          python3-vcstool \
          build-essential && \
    rosdep init && \
    rosdep update && \
    rm -rf /var/lib/apt/lists/*





RUN mkdir ros_catkin_ws && \
    cd ros_catkin_ws && \
    rosinstall_generator ${ROS_PKG} vision_msgs --rosdistro ${ROS_DISTRO} --deps --tar > ${ROS_DISTRO}-${ROS_PKG}.rosinstall && \
    mkdir src && \
    vcs import --input ${ROS_DISTRO}-${ROS_PKG}.rosinstall ./src && \
    apt-get update && \
    rosdep install --from-paths ./src --ignore-packages-from-source --rosdistro ${ROS_DISTRO} --skip-keys python3-pykdl -y && \
    python3 ./src/catkin/bin/catkin_make_isolated --install --install-space ${ROS_ROOT} -DCMAKE_BUILD_TYPE=Release && \
    rm -rf /var/lib/apt/lists/*


RUN pip3 install trimesh \
  numpy-quaternion \
  networkx \
  pyyaml \
  rospkg \
  rosdep \
  empy


ARG CACHE_DATE=2024-07-19




RUN pip3 install warp-lang



RUN mkdir /pkgs && cd /pkgs && git clone https:

ENV TORCH_CUDA_ARCH_LIST "7.0+PTX"

RUN cd /pkgs/curobo && pip3 install .[dev] --no-build-isolation

WORKDIR /pkgs/curobo


ENV  PYOPENGL_PLATFORM=egl


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
    sh install.sh $(python3 -c 'import torch.utils; print(torch.utils.cmake_prefix_path)') && \
    python3 -m pip install -e .


RUN python3 -m pip install "robometrics[evaluator] @ git+https://github.com/fishbotics/robometrics.git"


RUN python3 -m pip install typing-extensions --upgrade


RUN python3 -m pip install numpy --upgrade