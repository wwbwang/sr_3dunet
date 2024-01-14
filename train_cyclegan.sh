CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch\
 --nproc_per_node=2 --master_port=12342 sr_3dunet/train.py \
 -opt options/train_projection_cyclegan_rotatedVISoR.yml --launcher pytorch #  --auto_resume