CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun\
 --nproc_per_node=4 --master_port=12340 sr_3dunet/train.py \
 -opt options/MPCN_VISoR.yml --launcher pytorch # --auto_resume