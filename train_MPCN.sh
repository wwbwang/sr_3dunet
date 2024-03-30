CUDA_VISIBLE_DEVICES=0,1,2,5,6,7 python -m torch.distributed.launch\
 --nproc_per_node=6 --master_port=12342 sr_3dunet/train.py \
 -opt options/DPCN.yml --launcher pytorch # --auto_resume