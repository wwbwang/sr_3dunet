python train.py -model neuron_dev \
                -arch RESIN \
                -dataset tif_dataset \
                -trainer trainer \
                -gpus 0 \
                -batch_per_gpu 16 \
                -epochs 1000 \
                -save_every 5 \
                -out out \
                -cfg config/RESIN.yaml \
                -slurm \
                -slurm_ngpus 3 \
                -slurm_nnodes 1 \
                -slurm_nodelist c001 \
                -slurm_partition compute \
                -reset