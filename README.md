# HowTOs

***善用加粗、倾斜***

### 📖 **RESIN: A self-supervised framework for enhancing axial resolution of volumetric imaging data**
> [![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://baidu.com)<br>
> [Author](https://github.com)

Based on BasicSR.

### 🚩 Updates
* **2024.9.16**: XXX

## Web Demo and API

[![Replicate](https://replicate.com/cjwbw/animesr/badge)]() 
*coming soon*

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

## ⚡ **Inference**

We provided 3 ways to inference images, include inference **tif files in a given folder_path**, **tif files in a given folder_path**, **tif files in a given folder_path**:

讲清楚区别在于分开的原因在于norm方式

```bash
# inference tif files in folder
bash inference_from_folder.sh

# inference a single h5 file
bash inference_from_h5.sh
```

We also supply a simple scripts for you to quickly inference your 

## 💻 Training

### **Dataset Preparation**

1. Get a big ROI regin from your data, the format of your data should be h5/ims (or tif/tiff).
2. Crop the above big single image to sub-images, 128×128, for example. Note that you can apply filtering rules (using args arguments or write yourself's code) when selecting data subsets, as ensuring the cleanliness of the dataset is essential for successful training.
    
    The scripts is as follows:
    ```bash
    # Get sub-images from a single tifffile
    python scripts/get_subdatasets_from_bigtif.py --size=128 --percentiles_lower_bound=0.75 --percentiles_upper_bound=0.99 --minmax=450 --maxmin=60000 --input_folder="path" --output_folder="path" --input_tif_name="name" --crop_size=50
    # Get sub-images from a single h5file
    python scripts/get_subdatasets_from_ims.py --size=128 --percentiles_lower_bound=0 --percentiles_upper_bound=0.9999 --minmax=400 --maxmin=60000 --input_folder="path" --output_folder="path" --input_tif_name="name" --x_floor=0 --x_upper=8700 --y_floor=0 --y_upper=8400 --z_floor=0 --z_upper=900 --channel=1
    ```
    If the direction of anisotropy is not axial but 45° orientation, add `--rotated_flag` at the end of the command. The py_script will generate 2 (or 3 if rotated_flag is True) datasets, as `frontground_datasets`, `background_datasets` (and `rotated_frontground_datatsets`). Note that the `background_datasets` is just for check.

### **Start Training**
Let's take the de-anisotropy pipeline with VISoR's NISSL datasets for example.

支持neuron，nissl，dapi等
