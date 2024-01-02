
branch_name="projection_cyclegan"
iter='30200'
CUDA_VISIBLE_DEVICES=2 python scripts/inference_projection_cyclegan.py \
    --expname "${branch_name}_net_g_${iter}" --num_io_consumer 1 --iso_dimension -1\
    -i /home/wangwb/workspace/sr_3dunet/datasets/rotated_blocks/val\
    -o /home/wangwb/workspace/sr_3dunet/results/${branch_name}_net_g_${iter}\
    --model_path /home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_A_${iter}.pth\
    --model_back_path /home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_B_${iter}.pth
