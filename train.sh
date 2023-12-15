CUDA_VISIBLE_DEVICES=0,1,3 python -m torch.distributed.launch\
 --nproc_per_node=3 --master_port=12341 sr_3dunet/train.py \
 -opt options/train_unet_3d.yml --launcher pytorch # --auto_resume