
ARG ISAACLAB_BASE_IMAGE_ARG=isaac-lab-base


FROM ${ISAACLAB_BASE_IMAGE_ARG} AS base


SHELL ["/bin/bash", "-c"]

ARG ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG
ENV ISAACLAB_EXTENSION_TEMPLATE_PATH=${ISAACLAB_EXTENSION_TEMPLATE_PATH_ARG}


ARG ISAACLAB_PATH_ARG
ENV ISAACLAB_PATH=${ISAACLAB_PATH_ARG}


ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

USER root


COPY ../ ${ISAACLAB_EXTENSION_TEMPLATE_PATH}








RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc-11 g++-11 \
    python3.10-dev \
    git \
    wget \
    git-lfs \
    nano \
    tmux \
    pkg-config \
    zip \
    unzip \
    ca-certificates && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*








ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH


FROM nvcr.io/nvidia/cuda:12.4.0-devel-ubuntu22.04 AS cuda-devel




FROM base
COPY --from=cuda-devel /usr/local/cuda /usr/local/cuda


RUN echo "alias run=/isaac-sim/isaac-sim.streaming.sh" >> ${HOME}/.bashrc && \
    echo "alias lab=${ISAACLAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc


ENV LIVESTREAM=2


RUN /isaac-sim/python.sh -m pip install --upgrade pip && \
    rm /usr/bin/python3.10 && \
    ln -s /isaac-sim/kit/python/bin/python3.10 /usr/bin/python3.10










RUN bash -i -c "git config --global --add safe.directory /workspace/tacex"



RUN wget https:
    chmod +x cmake-3.26.4-linux-x86_64.sh && \
    mkdir /opt/cmake && \
    ./cmake-3.26.4-linux-x86_64.sh --prefix=/opt/cmake --skip-license && \
    rm cmake-3.26.4-linux-x86_64.sh
ENV PATH="/opt/cmake/bin:${PATH}"


RUN cd /opt && \
git clone https:
cd vcpkg && \
./bootstrap-vcpkg.sh -disableMetrics


ENV VCPKG_ROOT=/opt/vcpkg
ENV CMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake



WORKDIR ${ISAACLAB_EXTENSION_TEMPLATE_PATH}
