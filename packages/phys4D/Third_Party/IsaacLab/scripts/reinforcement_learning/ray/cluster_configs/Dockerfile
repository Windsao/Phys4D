FROM isaac-lab-base:latest



RUN apt-get update && apt-get install wget


ENV PATH="/usr/local/nvidia/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib64"


RUN ln -sf /usr/local/nvidia/bin/nvidia* /usr/bin


RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install "ray[default, tune]"==2.31.0 && \
sed -i "1i $(echo "
/isaac-sim/kit/python/bin/ray && ln -s /isaac-sim/kit/python/bin/ray /usr/local/bin/ray


RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install optuna bayesian-optimization


RUN /workspace/isaaclab/_isaac_sim/python.sh -m pip install mlflow
