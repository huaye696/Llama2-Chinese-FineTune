TOT_CUDA="0,1"  # 总共几个显卡，及显卡的编号
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"
export OMP_NUM_THREADS=3
DATA_PATH="../../BigFile/data/allData.csv"
OUTPUT_PATH="../../BigFile/save/checkpoint-2GPU"
MODEL_PATH="../../BigFile/model/Llama_Model"
SAVE_PATH="../../BigFile/save/model-save-2GPU"
EPOCHS=1
VAL_SIZE=0.3

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--save_path $SAVE_PATH \
--val_size $VAL_SIZE \
--num_train_epochs $EPOCHS