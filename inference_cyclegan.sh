
branch_name="just_ssim"
iter='10000000000'  # nan Appoint in terminal

# 从命令行获取参数
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --iter)
            iter="$2"
            shift
            shift
            ;;
        *)
            echo "Unknown option: $key"
            exit 1
            ;;
    esac
done

CUDA_VISIBLE_DEVICES=2 python scripts/inference_projection_cyclegan.py \
    --expname "${branch_name}_net_g_${iter}" --num_io_consumer 1 --iso_dimension -1\
    -i /home/wangwb/workspace/sr_3dunet/datasets/rotated_blocks/val\
    -o /home/wangwb/workspace/sr_3dunet/results/${branch_name}_net_g_${iter}\
    --model_path /home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_A_${iter}.pth\
    --model_back_path /home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_B_${iter}.pth
    # --model_path /home/wangwb/workspace/sr_3dunet/weights/projection_cyclegan_netg_A_${iter}.pth\
    # --model_back_path /home/wangwb/workspace/sr_3dunet/weights/projection_cyclegan_netg_B_${iter}.pth
