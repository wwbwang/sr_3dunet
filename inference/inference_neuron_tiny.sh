# set task name of this data
TASK_NAME="neuron_tiny"

CFG_PATH="config/${TASK_NAME}.yaml"

# dir or file
IMG_PATH="data/RESIN/neuron/val/little_dirty.tif"

DEBUG=false
DEBUG=true
SAVE_PATH="inference/result/${TASK_NAME}"
if [ "${DEBUG}" = true ]; then
    SAVE_PATH="inference/debug/${TASK_NAME}"
fi

# set epoch of the checkpoint
EPOCH=200
EPOCH=$(printf "%04d" $EPOCH)
CKPT_PATH="out/weights/${TASK_NAME}/Epoch_${EPOCH}.pth"

CMD="python inference.py -cfg ${CFG_PATH} \
                         -ckpt_path ${CKPT_PATH} \
                         -img_path ${IMG_PATH} \
                         -save_path ${SAVE_PATH} \
                         -piece_size 64 \
                         -overlap 16"

# Add the -debug flag if in debug mode
if [ "${DEBUG}" = true ]; then
    CMD="$CMD -debug"
fi

# Execute the command
eval $CMD