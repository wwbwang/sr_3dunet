# Install

```
git clone https://github.com/wwbwang/sr_3dunet.git -b dev
cd sr_3dunet/
```

# Inference

- single, example :

  ```
  sh inference/inference_script/infer_single_nissl.sh
  ```

  ```
  #!/bin/bash
  python inference/inference_single.py -cfg config/RESIN_nissl.yaml \
                                       -ckpt_path out/weights_GA/nissl_fullDS/Epoch_0080.pth \
                                       -img_path data/RESIN/nissl/val/standard_128.tif \
                                       -save_dir inference/result/nissl \
                                       -save_name RESIN_output.tif \

  ```
- batch, example :

  ```
  sh inference/inference_script/infer_batch_nissl.sh
  ```

  ```
  #!/bin/bash
  python inference/inference_batch.py -cfg config/RESIN_nissl.yaml \
                                      -ckpt_path out/weights_GA/nissl_fullDS/Epoch_0080.pth \
                                      -img_dir data/RESIN/nissl/val \
                                      -save_dir inference/result/nissl \

  ```
