CUDA_VISIBLE_DEVICES=3 python -m torch.distributed.launch\
 --nproc_per_node=1 --master_port=12342 sr_3dunet/train.py \
 -opt options/train_projection_cyclegan.yml --launcher pytorch # --auto_resume