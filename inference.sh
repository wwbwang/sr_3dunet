
branch_name="test0_archived_20231215_145350"
iter='10000'
CUDA_VISIBLE_DEVICES=2 python scripts/inference.py \
    --expname "${branch_name}_net_g_${iter}" --num_io_consumer 1\
    -i /home/wangwb/workspace/sr_3dunet/datasets/rotated_blocks/val\
    -o /home/wangwb/workspace/sr_3dunet/results/${branch_name}_net_g_${iter}\
    --model_path /home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_${iter}.pth
