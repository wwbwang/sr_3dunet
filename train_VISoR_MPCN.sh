CUDA_VISIBLE_DEVICES=0,1 torchrun\
 --nproc_per_node=2 --master_port=12340 sr_3dunet/train.py \
 -opt options/MPCN_VISoR_noA2C.yml --launcher pytorch # --auto_resume


 # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=12340 sr_3dunet/train.py  -opt options/MPCN_VISoR_noA2C.yml --launcher pytorch