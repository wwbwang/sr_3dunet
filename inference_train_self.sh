branch_name="train_self_rotated"
iter='10000000000'

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


CUDA_VISIBLE_DEVICES=2 python scripts/inference_train_self.py \
    --expname "${branch_name}_net_g_${iter}" --num_io_consumer 1\
    -i /home/wangwb/workspace/sr_3dunet/datasets/rotated_blocks/val\
    -o /home/wangwb/workspace/sr_3dunet/results/${branch_name}_net_g_${iter}\
    --model_path /home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_${iter}.pth