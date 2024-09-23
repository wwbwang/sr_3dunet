# HowTOs

### 📖 **RESIN: A self-supervised framework for enhancing axial resolution of volumetric imaging data**
> [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://baidu.com)<br>
> [Author](https://github.com)

### 🚩 Updates
* **2024.9.24**: 【修改说明】补充README中的训练可视化
* **2024.9.24**: 【修改说明】bug已修复
* **2024.9.24**: 【修改说明】README已上传，bug待修理
* **2024.9.23**: 【修改说明】代码整改完毕，README待上传，bug待修理


## Web Demo and API

[![Replicate](https://replicate.com/cjwbw/animesr/badge)]() 

The web demo and API (Huggingface) is *coming soon*.

## Video Demos

*Coming soon*

## 🔧 Dependencies and Installation
- Python >= 3.7 (Recommend to use Anaconda or Miniconda)
- PyTorch >= 1.7
- Other required packages in requirements.txt

### **Installation**

1. Clone the repo.

    ```bash
    git clone https://github.com/wwbwang/sr_3dunet
    cd sr_3dunet
    ```
2. Install the dependent packages and RESIN.

    ```bash
    # Install dependent packages
    python==3.9.0
    pip install basicsr
    pip install -r requirements.txt

    # Install RESIN
    python setup.py develop
    ```

3. Switch Branch to RESIN.
    ```bash
    git checkout RESIN
    ```

## 💻 Training

### **Overview**

The training pipeline contains 2 steps, Dataset Preparation and Training. The training params is in `./options`，you can create yourself's yml file for your training pipeline. More params' explanation can be found in [here](https://github.com/XPixelGroup/BasicSR-docs).

### **Dataset Preparation**

1. Get a big ROI regin from your data, the format of the chosen data should be h5/ims (or tif/tiff).
2. Crop the above big single image to sub-images TIFFs, size 128×128, for example. Note that you can apply filtering rules such as filtering background or low-quality data. The purpsoe of this step is ensuring the cleanliness of the training dataset.

    Then The scripts is as follows:
    ```bash
    # Get sub-images from a single tifffile
    python scripts/get_subdatasets_from_bigtif.py --size=128 --percentiles_lower_bound=0.75 --percentiles_upper_bound=0.99 --minmax=450 --maxmin=60000 --input_folder="path" --output_folder="path" --input_tif_name="name" --crop_size=50 --rotated_flag=False

    # Get sub-images from a single h5file
    python scripts/get_subdatasets_from_ims.py --size=128 --percentiles_lower_bound=0 --percentiles_upper_bound=0.9999 --minmax=400 --maxmin=60000 --input_folder="path" --output_folder="path" --input_tif_name="name" --x_floor=0 --x_upper=8700 --y_floor=0 --y_upper=8400 --z_floor=0 --z_upper=900 --channel=1 --rotated_flag=False
    ```

    The py_script will generate 2 (or 3 if rotated_flag is True) datasets, as `frontground_datasets`, `background_datasets` (and `rotated_frontground_datatsets`). Note that the `background_datasets` is just for check. If your params setting is good, all dirty data should should be in the `background_datasets` folder, and all the `background_datasets` folder's data should be dirty.

### **Training**

Before the training, you should modify the [yaml file](options/RESIN.yml) accordingly. For example, you should modify the `datasets_cube` to your own anisotropic dataset. The training instruction is as follows:

```bash
# Using slurm
CUDA_VISIBLE_DEVICES=0,1,2,3 srun\
--ntasks-per-node=4 --mpi=pmi2 python sr_3dunet/train.py \
-opt options/RESIN.yml --launcher="slurm" # --auto_resume
```

### **Training Visualization**

1. You can visualize your training loss using `Tensorboard` as follows:
    ```bash
    tensorboard --logdir ./tb_logger
    ```

2. You can use `wandb` to visualize training status as well. About wandb, please refer to the [Documentation of BasicSR](https://github.com/XPixelGroup/BasicSR-docs).


## ⚡ **Inference**

RESIN support both TIFFs and ims/h5 as input for inference. We have provided corresponding scripts. Otherwise, we supply a debug inference script, for developers check training results quickly.

You can execute them in following manner:

**Inference TIFFs**

The script supports batch-inference for TIFF files in the given specified path. If your images are too large to fit in the GPU memory (it is common in 3D inference), you can set the `--piece*` parameter to process the images in samll pieces.

For some images with a slanted blur direction (like VISoR), if your model is not trained with slanted orientations, you can set the 'rotated_flag' parameter. The script will automatically inference in rotated manner.

```bash
# bash inference_from_folderTIFFs.sh
CUDA_VISIBLE_DEVICES=0 python scripts/inference_from_folder.py \
    -i /share/home/wangwb/workspace/sr_3dunet/datasets/rm009_labeled/bg \
    -o /share/home/wangwb/workspace/sr_3dunet/datasets/rm009_labeled/bg_sr \
    --model_path /share/home/wangwb/workspace/sr_3dunet/weights/MPCN_VISoR_oldbaseline_256_net_g_A_140000.pth \
    --piece_flag True --piece_size 128 --piece_overlap 16 --rotated_flag False
```

```console
Usage:
    -i, --input:        The folder path containing your test TIFF files.
    -o, --output:       The folder path for the model's output TIFF files; this folder may not already exist.
    --model_path:       The path of your Restoration model file (.pth).
    --piece_flag:       Set to True if you wants to processing large TIFF files by splitting them into smaller segments before inference.
    --piece_size:       Applicable when "--piece_flag" is enabled, defines the dimensions of the smaller TIFF segments.
    --piece_overlap:    Applicable when "--piece_flag" is enabled, indicates the overlap area between adjacent smaller TIFF segments.
    --rotated_flag:     Set to True if your model expects horizontal data but the test data contains oblique angles (e.g., in VISoR).
```

**Inference ims/h5**

Different from TIFFs-Inference script, the Ims/H5-Inference scripts only supports single file inference and no rotated data, cause the Ims/H5 files are usually single and big. If you wants to do batch inference, you can write inference scripts yourself.

```bash
# bash inference_from_h5.sh
CUDA_VISIBLE_DEVICES=0 python scripts/inference_from_h5.py \
    -i /share/data/VISoR_Reconstruction/SIAT_ION/LiuCiRong/20230910_CJ004/CJ4-1um-ROI1/CJ4ROI2.ims \
    -o datasets/NISSL/CJ4ROI2_out_h5_piece64/output_res0.h5 \
    --model_path weights/MPCN_VISoR_NISSL_net_g_A_110000.pth \
    --piece_size 64 --piece_overlap 16
```

```console
Usage:
    -i, --input:        Your test h5/ims file's path.
    -o, --output:       Your model\'s output h5/ims file's path.
    --model_path:       The path of your Restoration model file (.pth).
    --piece_size:       Defines the dimensions of the smaller image segments.
    --piece_overlap:    Indicates the overlap area between adjacent smaller image segments.
```

**Inference for debug**

This scripts is written for developers and users to check training results quickly, so this scripts's parameters are more than above. It will create a new directory, and put input/output/reconstruct data in it. The `--piece*` parameters is setting as above. Otherwise, you can set `--remove_size` params to cut the specified most edge pixels, cause the `conv` operation is usually performs bad in image's edge, which obstructed practical visualization.

```bash
# bash inference_debug.sh
branch_names=("RESIN_concat_40X")
iters=("110000")

for branch_name in "${branch_names[@]}"
do
    for iter in "${iters[@]}"
    do 
        CUDA_VISIBLE_DEVICES=0 python scripts/inference_debug.py \
            -i /share/home/wangwb/workspace/sr_3dunet/datasets/40X/val_datasets\
            -o /share/home/wangwb/workspace/sr_3dunet/results/${branch_name}_net_g_${iter}\
            --model_path /share/home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_A_${iter}.pth\
            --model_back_path /share/home/wangwb/workspace/sr_3dunet/experiments/${branch_name}/models/net_g_B_${iter}.pth\
            --piece_flag True --piece_size 64 --piece_overlap 16 --remove_size 4 --rotated_flag True
    done
done
```

```console
Usage:
    -i, --input:        The folder path containing your test TIFF files.
    -o, --output:       The folder path for the model's output TIFF files; this folder may not already exist.
    --model_path:       The path of your Restoration model file (.pth).
    --model_back_path:  The path of the simulated degradation model file obtained through cycle-consistency training, which aids in visualizing more training results.
    --piece_flag:       Set to True if you wants to processing large TIFF files by splitting them into smaller segments before inference.
    --piece_size:       Applicable when "--piece_flag" is enabled, defines the dimensions of the smaller TIFF segments.
    --piece_overlap:    Applicable when "--piece_flag" is enabled, indicates the overlap area between adjacent smaller TIFF segments.
    --remove_size:      The number of pixels to be trimmed from the outermost edge.
    --rotated_flag:     Set to True if your model expects horizontal data but the test data contains oblique angles (e.g., in VISoR).
```
After run the above command, you will get the de-anisotropy images in specified output path.

## Acknowledgement
This project is build based on BasicSR.

##  Citation
If you find this project useful for your research, please consider citing our paper:
```bibtex
@InProceedings{wang2024RESIN,
  author={XXX},
  title={RESIN: A self-supervised framework for enhancing axial resolution of volumetric imaging data},
  booktitle={XXX},
  year={2024}
}
```

## 📧 Contact
If you have any question, please email `wwbwang99@gmail.com`.
