CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch\
 --nproc_per_node=2 --master_port=12342 sr_3dunet/train.py \
 -opt options/train_projection_cyclegan.yml --launcher pytorch # --auto_resume