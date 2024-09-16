## HowTOs

### ✨**Installation**

1. Clone repo

    ```bash
    git clone https://github.com/wwbwang/sr_3dunet
    cd sr_3dunet
    ```
2. Install dependent packages and RESIN

    ```bash
    # Install dependent packages
    python==3.9.0
    pip install basicsr
    pip install -r requirements.txt

    # Install sr_3dunet
    python setup.py develop
    ```

3. Switch Branch to RESIN
    ```bash
    git checkout RESIN
    ```

### ✨**Start training**
We take the de-anisotropy pipeline with VISoR's NISSL datasets for example.

#### **Axial anisotropy**
1. Get a big ROI regin from your data, save it to h5 (or tiff).
2. Crop the above big single data to sub-images, 128×128, for example. The training patch is usually small, and get these patch should has a rule. Note that you can set some percentile rules during selecting these sub-images, and the size of this sub-image is different from the traing patch_size, the dataloader will furter randomly crop the sub-images to GT_size×GT_size patchs for training.
Run the script extract_subimages.py:
    ```bash
    # 
    bash scripts/
    ```


支持neuron，nissl，dapi等

### ✨**Inference**

```bash
# inference tif files in folder
bash inference_from_folder.sh

# inference a single h5 file
bash inference_from_h5.sh
```