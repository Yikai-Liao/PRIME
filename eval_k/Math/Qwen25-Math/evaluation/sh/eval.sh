set -ex

PROMPT_TYPE=$1
MODEL_NAME_OR_PATH=$2
OUTPUT_DIR=$3
TEST_MODE=$4
TEMPERATURE=${5:-0.8}  # Default to 0.8 if not provided
SEED=${6:-42}          # Default to 42 if not provided

SPLIT="test"
# Set NUM_TEST_SAMPLE and N_SAMPLING based on test mode
if [ "$TEST_MODE" = "true" ]; then
    NUM_TEST_SAMPLE=10
    N_SAMPLING=10
    echo "Test mode: Evaluating only first 10 problems with 10 samples each"
else
    NUM_TEST_SAMPLE=-1
    N_SAMPLING=10
fi

# English open datasets
DATA_NAME="minerva_math,olympiadbench"
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed ${SEED} \
    --temperature ${TEMPERATURE} \
    --n_sampling ${N_SAMPLING} \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --apply_chat_template \









