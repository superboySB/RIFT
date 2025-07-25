# 笔记
## 配置
```sh
docker build -t dzp_carla:0714 --network=host --progress=plain .

xhost +

docker run -itd --privileged --gpus all --net=host \
  -v /home/matt/projects/RIFT /workspace/RIFT \
  -e DISPLAY=$DISPLAY \
  -e SDL_AUDIODRIVER=dummy \
  --name dzp-carla-0714 \
  dzp_carla:0714 \
  /bin/bash

docker exec -it dzp-carla-0714 /bin/bash

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

## 复现实验
### 研究数据对接
```sh
python analysis_npz_map_data.py
```

### 研究训练过程
时不时会崩溃
```sh
bash scripts/run_multi.sh -t 3 -e pdm_lite.yaml -c rift_pluto.yaml -m train_cbv -r 2 -s 0 -g 0
```

### 研究推理过程
```sh
CUDA_VISIBLE_DEVICES=0 python scripts/run.py --ego_cfg plant.yaml --cbv_cfg rift_pluto.yaml --mode eval -rep 1
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