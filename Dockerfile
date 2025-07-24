FROM carlasim/carla:0.9.15

LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

ENV DEBIAN_FRONTEND=noninteractive

# 如果需要走代理
ENV http_proxy=http://127.0.0.1:8889
ENV https_proxy=http://127.0.0.1:8889

# 用root身份安装依赖
USER root

# 安装 Python3.8 和常用工具
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    git tmux vim gedit curl sudo && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-distutils python3.8-dev unzip sudo python3-tk ffmpeg

# 切换 python3 默认到 python3.8
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2 && \
    update-alternatives --set python3 /usr/bin/python3.8 && \
    ln -sfn /usr/bin/python3.8 /usr/bin/python

# 安装 pip3.8
RUN curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python3.8

# 给 carla 用户 sudo 权限+免密码
RUN usermod -aG sudo carla && \
    echo "carla ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /home/carla
RUN cd Import && apt-get install -y wget && \
    wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/AdditionalMaps_0.9.15.tar.gz
RUN bash /home/carla/ImportAssets.sh

# 切换回 carla 用户
USER carla

# 环境变量
ENV GLOG_minloglevel=2
ENV MAGNUM_LOG="quiet"

# 配置 Python路径
RUN echo 'export CARLA_ROOT=/home/carla' >> ~/.bashrc && \
    echo 'export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI' >> ~/.bashrc && \
    echo 'export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla' >> ~/.bashrc && \
    echo 'export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg' >> ~/.bashrc

# 工作目录
WORKDIR /workspace
RUN git clone https://github.com/superboySB/RIFT && cd RIFT && \
    python3 -m pip install --retries=10 --timeout=120 --no-cache-dir -r requirements.txt
RUN cd RIFT && python3 -m pip install .

# 如需清理代理，取消注释
# ENV http_proxy=
# ENV https_proxy=
# ENV no_proxy=
# RUN rm -rf /var/lib/apt/lists/* && apt-get clean

CMD ["/bin/bash"]
