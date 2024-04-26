CUDA_VISIBLE_DEVICES=0,1 torchrun\
 --nproc_per_node=2 --master_port=12340 sr_3dunet/train.py \
 -opt options/MPCN_VISoR_noA2C.yml --launcher pytorch # --auto_resume


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch\
    --nproc_per_node=4 --master_port=12340 sr_3dunet/train.py\
    -opt options/MPCN.yml --launcher pytorch

CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch\
    --nproc_per_node=4 --master_port=12341 sr_3dunet/train.py\
    -opt options/MPCN_VISoR.yml --launcher pytorch