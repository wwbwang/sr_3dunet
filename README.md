### Installation

1. Clone repo

    ```bash
    git clone https://github.com/wwbwang/sr_3dunet/tree/stable_version
    cd sr_3dunet
    ```
2. Install

    ```bash
    # Install dependent packages
    pip install -r requirements.txt

    # Install sr_3dunet
    python setup.py develop
    ```
    
**Inference**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/inference.py \
    --expname "base_ver" --num_io_consumer 1\
    -i /home/wangwb/workspace/sr_3dunet/datasets/rotated_blocks/val_rotated_small\
    -o /home/wangwb/workspace/sr_3dunet/results/stable_res\
    --model_path /home/wangwb/workspace/sr_3dunet/weights/net_g_A_70000.pth\
    --piece_flag True --piece_size 128 --overlap 16 --step_size 16 --rotated_flag True
```