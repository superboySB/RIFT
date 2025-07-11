# !/bin/bash
# 差一些对buffer的聚合过程，需要修改源码实现
for gpu_id in {0..7}; do
    echo "Starting GPU $gpu_id data collection..."
    CUDA_VISIBLE_DEVICES=$gpu_id bash scripts/run_multi.sh \
        -t 5 \
        -e pdm_lite.yaml \
        -c rift_pluto.yaml \
        -m train_cbv \
        -r 2 \
        -s $gpu_id \
        -g $gpu_id &
done
wait
echo "All data collection completed!"