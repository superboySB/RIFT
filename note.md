# 笔记
## 配置
```sh
docker build -t dzp_carla:test --network=host --progress=plain .

xhost +

docker run -itd --privileged --gpus all --net=host \
  -e DISPLAY=$DISPLAY \
  -e SDL_AUDIODRIVER=dummy \
  --name dzp-carla-test \
  dzp_carla:test \
  /bin/bash

docker exec -it dzp-carla-test /bin/bash

cd ~ && bash ./CarlaUE4.sh
```
测试carla功能正常即可，然后下载`README.md`里面提供的四个zip，移动外部文件进来
```sh
docker cp ~/Downloads/for_RIFT_20250714 dzp-carla-test:/workspace/
```
最好重新删掉然后安装一下RIFT(`python3 -m pip install .`)便于版本管理，整理directory
```sh
unzip /workspace/for_RIFT_20250714/PlanT_medium-20250711T161141Z-1-001.zip -d /workspace/RIFT/rift/ego/model_ckpt/ && \
unzip /workspace/for_RIFT_20250714/pluto-20250711T161146Z-1-001.zip -d /workspace/RIFT/rift/cbv/planning/model_ckpt/ && \
unzip -j /workspace/for_RIFT_20250714/HD-Map-20250714T063845Z-1-001.zip -d /workspace/RIFT/data/map_data/ && \
unzip -j /workspace/for_RIFT_20250714/Speed-Limits-20250711T161132Z-1-001.zip -d /workspace/RIFT/data/speed_limits/
```
运行验证方式参考`README.md`

## 研究数据对接
```sh
python data/extract_complete_npz_data.py
```
## QA

### 便于递交IT
```sh
docker commit [容器ID或名字] [新镜像名:标签] # 如dzp_carla:0715
docker save 
docker load
```
### 换源
```sh
sudo cp /etc/apt/sources.list /etc/apt/sources.list.bak
sudo vim /etc/apt/sources.list
```
写进去
```bash
deb http://mirrors.cowarobot.cn/ubuntu jammy main restricted universe multiverse
deb http://mirrors.cowarobot.cn/ubuntu jammy-updates main restricted universe multiverse
deb http://mirrors.cowarobot.cn/ubuntu jammy-backports main restricted universe multiverse
deb http://mirrors.cowarobot.cn/ubuntu jammy-security main restricted universe multiverse
```
然后重新
```sh
sudo apt-get update
```
### 换代理
```sh
unset https_proxy && unset http_proxy
```