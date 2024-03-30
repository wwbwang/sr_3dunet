# branch_name=MPCN_0.9999_simulation_Diter2
branch_name=MPCN_baseline
# branch_name=stepnet_patch64_d4g2_no_norm!!save
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

# rotated_blocks/val_rotated_big\
CUDA_VISIBLE_DEVICES=0 python scripts/inference_pretrained.py \
    --expname "${branch_name}_net_g_${iter}" --num_io_consumer 1\
    -i /home/wangwb/datasets/simulation_newselfnet/blurred/blurred_split\
    -o /home/wangwb/workspace/sr_3dunet/results/${branch_name}_net_g_${iter}\
    --model_path /home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_A_${iter}.pth\
    --model_back_path /home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_B_${iter}.pth\
    --piece_flag True --piece_size 128 --overlap 0 --step_size 16 --rotated_flag False

    # /home/wangwb/workspace/sr_3dunet/datasets/rotated_blocks/val_
    # /home/wangwb/datasets/simulation_newselfnet/blurred/blurred_split    
    # /home/wangwb/workspace/sr_3dunet/datasets/simulation_newselfnet/blurred/val\
    # /home/wangwb/workspace/sr_3dunet/datasets/val\
    # /home/wangwb/workspace/sr_3dunet/datasets/rotated_blocks/val_rotated_small\
    # /home/wangwb/workspace/sr_3dunet/datasets/val_simulation_selfnet\