TOT_CUDA="0,1"  # 总共几个显卡，及显卡的编号
CUDAs=(${TOT_CUDA//,/ })
CUDA_NUM=${#CUDAs[@]}
PORT="12345"

DATA_PATH="allData.csv"
OUTPUT_PATH="../chinese/llama-7b-lwh-3"
MODEL_PATH="../model/Llama_Model"
SAVE_PATH="../chinese/llama-7b-save"
VAL_SIZE=0.3

CUDA_VISIBLE_DEVICES=${TOT_CUDA} torchrun --nproc_per_node=$CUDA_NUM --master_port=$PORT finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--eval_steps 200 \
--save_steps 2000 \
--save_path $SAVE_PATH \
--val_size $VAL_SIZE