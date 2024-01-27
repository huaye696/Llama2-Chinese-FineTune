export OMP_NUM_THREADS=3
DATA_PATH="../../BigFile/data/test.csv"
OUTPUT_PATH="../../BigFile/save/checkpoint"
MODEL_PATH="../../BigFile/model/Llama_Model"
SAVE_PATH="../../BigFile/save/model-save"
VAL_SIZE=0.3

CUDA_VISIBLE_DEVICES="1" python finetune.py \
--data_path $DATA_PATH \
--output_path $OUTPUT_PATH \
--model_path $MODEL_PATH \
--save_path $SAVE_PATH \
--val_size $VAL_SIZE