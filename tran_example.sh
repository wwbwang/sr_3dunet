python train.py -model INRs \
                -arch mlp \
                -dataset random \
                -trainer trainer \
                -gpus 0 \
                -batch_per_gpu 200 \
                -epochs 100 \
                -save_every 5 \
                -out out \
                -cfg config/example.yaml \
                -reset \
                # -slurm \
                # -slurm_ngpus 4 \
                # -slurm_nnodes 1 \
                # -slurm_nodelist c003 \
                # -slurm_partition compute \