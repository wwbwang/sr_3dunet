#!/bin/bash
branch_names=("!!saveMPCN_VISoR_oldbaseline_256")
iters=("14000")

# iters=()
# for ((i=20000; i<=160000; i+=20000))
# do
#     iters+=("$(echo $i)")
# done

for branch_name in "${branch_names[@]}"
do
    for iter in "${iters[@]}"
    do 
        CUDA_VISIBLE_DEVICES=0 python scripts/inference_bk.py \
            --expname "${branch_name}_net_g_${iter}" --num_io_consumer 1\
            -i /share/home/wangwb/workspace/sr_3dunet/datasets/Monkey_Brain/val_datasets\
            -o /share/home/wangwb/workspace/sr_3dunet/results/${branch_name}_net_g_${iter}\
            --model_path /share/home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_A_${iter}.pth\
            --model_back_path /share/home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_B_${iter}.pth\
            --piece_flag False --piece_size 128 --piece_overlap 16 --piece_mod_size 16 --remove_size 0 --rotated_flag False
    done
done

    # neuron 
    # !!saveMPCN_VISoR_oldbaseline_256
    # !!saveMPCN_VISoR_NISSL
    # MPCN_VISoR_NISSL_percentiles01_sqrt_absborm
    # /share/home/wangwb/workspace/sr_3dunet/datasets/Monkey_Brain/val_datasets
    # /share/home/wangwb/workspace/sr_3dunet/datasets/NISSL/val_standard_datasets

    # NISSL
    # /share/home/wangwb/workspace/sr_3dunet/datasets/Monkey_Brain/skels_rm009
    # /share/home/wangwb/workspace/sr_3dunet/datasets/Mouse_Brain/val_datasets
    # /share/home/wangwb/workspace/sr_3dunet/datasets/bigtif
    # /share/home/wangwb/workspace/sr_3dunet/datasets/origin_blocks
    # /share/home/wangwb/workspace/sr_3dunet/datasets/test_tifs
    # /share/home/wangwb/workspace/sr_3dunet/datasets/rotated_LGN-V1-ROI/rotated_datasets


# # 从命令行获取参数
# while [[ $# -gt 0 ]]; do
#     key="$1"
#     case $key in
#         --iter)
#             iter="$2"
#             shift
#             shift
#             ;;
#         *)
#             echo "Unknown option: $key"
#             exit 1
#             ;;
#     esac
# done
