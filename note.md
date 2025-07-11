笔记
```sh
docker build -t dzp_carla:test --network=host --progress=plain .

xhost +

docker run -itd --privileged --gpus all --net=host \
  -e DISPLAY=$DISPLAY \
  -e SDL_AUDIODRIVER=dummy \
  dzp_carla:test  dzp-carla-test \
  /bin/bash

docker exec -it dzp-carla-test /bin/bash

bash ./CarlaUE4.sh
```