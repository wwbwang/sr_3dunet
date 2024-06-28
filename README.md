### Installation

1. Clone repo

    ```bash
    git clone https://github.com/wwbwang/sr_3dunet
    cd sr_3dunet
    ```
2. Install

    ```bash
    # Install dependent packages
    python==3.9.0
    pip install basicsr
    pip install -r requirements.txt

    # Install sr_3dunet
    python setup.py develop
    ```
    
**Inference**

```bash
# inference tif files in folder
bash inference_from_folder.sh

# inference a single h5 file
bash inference_from_h5.sh
```