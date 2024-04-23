branch_name=SIREN_noBN_noscreen_print1_
# branch_name=MPCN_VISoR_new_datasets
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

CUDA_VISIBLE_DEVICES=0 python scripts/inference_pretrained.py \
    --expname "${branch_name}_net_g_${iter}" --num_io_consumer 1\
    -i /share/home/wangwb/workspace/sr_3dunet/datasets/Mouse_Brain/val_datasets\
    -o /share/home/wangwb/workspace/sr_3dunet/results/${branch_name}_net_g_${iter}\
    --model_path /share/home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_A_${iter}.pth\
    --model_back_path /share/home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_B_${iter}.pth\
    --piece_flag True --piece_size 128 --piece_overlap 16 --piece_mod_size 16 --remove_size 8 --rotated_flag False

    # /share/home/wangwb/workspace/sr_3dunet/datasets/Mouse_Brain/val_datasets
    # /share/home/wangwb/workspace/sr_3dunet/datasets/bigtif
    # /share/home/wangwb/workspace/sr_3dunet/datasets/origin_blocks
    # /share/home/wangwb/workspace/sr_3dunet/datasets/test_tifs
    # /share/home/wangwb/workspace/sr_3dunet/datasets/rotated_LGN-V1-ROI/rotated_datasets