











ARG IMAGE_TAG
FROM curobo_docker:${IMAGE_TAG}

ARG USERNAME
ARG USER_ID
ARG CACHE_DATE=2024-07-19






RUN useradd -l -u $USER_ID -g users $USERNAME

RUN /sbin/adduser $USERNAME sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN usermod -aG root $USERNAME



RUN mkdir /isaac-sim/kit/cache && chown -R $USERNAME:users /isaac-sim/kit/cache
RUN chown $USERNAME:users /root && chown $USERNAME:users /isaac-sim
RUN mkdir /root/.nv && chown -R $USERNAME:users /root/.nv
RUN chown -R $USERNAME:users /root/.cache


RUN mkdir -p /isaac-sim/kit/logs/Kit/Isaac-Sim && chown -R $USERNAME:users /isaac-sim/kit/logs/Kit/Isaac-Sim




RUN mkdir /root/.nvidia-omniverse/logs && mkdir -p /home/$USERNAME/.nvidia-omniverse && cp -r /root/.nvidia-omniverse/* /home/$USERNAME/.nvidia-omniverse && chown -R $USERNAME:users /home/$USERNAME/.nvidia-omniverse
RUN chown -R $USERNAME:users /isaac-sim/exts/omni.isaac.synthetic_recorder/
RUN chown -R $USERNAME:users /isaac-sim/kit/exts/omni.gpu_foundation
RUN mkdir -p /home/$USERNAME/.cache && cp -r /root/.cache/* /home/$USERNAME/.cache && chown -R $USERNAME:users /home/$USERNAME/.cache
RUN mkdir -p /isaac-sim/kit/data/documents/Kit && mkdir -p /isaac-sim/kit/data/documents/Kit/apps/Isaac-Sim/scripts/ &&chown -R $USERNAME:users /isaac-sim/kit/data/documents/Kit /isaac-sim/kit/data/documents/Kit/apps/Isaac-Sim/scripts/
RUN mkdir -p /home/$USERNAME/.local


RUN echo "alias omni_python='/isaac-sim/python.sh'" >> /home/$USERNAME/.bashrc
RUN echo "alias python='/isaac-sim/python.sh'" >> /home/$USERNAME/.bashrc

RUN chown -R $USERNAME:users /home/$USERNAME




USER $USERNAME
WORKDIR /home/$USERNAME
ENV USER=$USERNAME
ENV PATH="${PATH}:/home/${USER}/.local/bin"
ENV SHELL /bin/bash
ENV OMNI_USER=admin
ENV OMNI_PASS=admin


RUN mkdir /root/Documents && chown -R $USERNAME:users /root/Documents

RUN echo 'completed'
