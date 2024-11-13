TASK_NAME="test"

# if need timestamp, set true. default is false
ADD_TIMESTAMP=false
TASK_NAME_ID="${TASK_NAME}"
if [ "${ADD_TIMESTAMP}" = true ]; then
    TIMESTAMP=$(date +"%Y%m%d_%H%M")
    TASK_NAME_ID="${TASK_NAME}_${TIMESTAMP}"
fi

# set config yaml file path
CFG_PATH="script/config.yaml"
CFG_BACKUP_PATH="config/${TASK_NAME_ID}.yaml"
if [ ${CFG_PATH} != ${CFG_BACKUP_PATH} ]; then
    cp ${CFG_PATH} ${CFG_BACKUP_PATH}
fi

# overwrite the directory with the same name
RESET=true

CMD="python train.py -model ${TASK_NAME_ID} \
                     -cfg ${CFG_PATH} \
                     -out out \
                     -gpus 0 \
                     -batch_per_gpu 4 \
                     -epochs 1000 \
                     -save_every 10 \
                     -slurm \
                     -slurm_ngpus 8 \
                     -slurm_nnodes 1 \
                     -slurm_nodelist c001 \
                     -slurm_partition compute"

if [ "$RESET" = true ]; then
    CMD="$CMD -reset"
fi

# Execute the command
eval $CMD