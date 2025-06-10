#!/bin/bash
source /home/yikai003/condaforge3/etc/profile.d/conda.sh   # set to your path
MODEL_CKPT=$1
NUM_SAMPLES=${2:-10}  # Number of samples per task (default: 10)
TEMPERATURE=${3:-0.8}  # Temperature for sampling (default: 0.8)
MODEL_NAME=$(basename "$MODEL_CKPT")
OUTPUT_DIR="results/$MODEL_NAME" # output dir
mkdir -p $OUTPUT_DIR

echo "Running evaluations with:"
echo "  Model: $MODEL_CKPT"
echo "  Samples per task: $NUM_SAMPLES"
echo "  Temperature: $TEMPERATURE"
echo "  Output directory: $OUTPUT_DIR"
echo


# Specify the test data set
# my_array=(mbpp leetcode math500 amc aime qwen livecodebench)
my_array=(amc)


if [[ " ${my_array[@]} " =~ " humaneval " ]]; then
    conda activate prime
    echo "running humaneval"
    mkdir -p $OUTPUT_DIR/human_eval_chat
    python3 Coding/human_eval/evaluate_human_eval.py \
        --model $MODEL_CKPT \
        --data_dir data/humaneval \
        --save_dir $OUTPUT_DIR/human_eval_chat \
        --num-samples-per-task $NUM_SAMPLES \
        
fi

if [[ " ${my_array[@]} " =~ " mbpp " ]]; then
    conda activate prime
    echo "running mbpp"
    mkdir -p $OUTPUT_DIR/mbpp_chat
    python3 -u Coding/mbpp/evaluate_mbpp.py \
        --model $MODEL_CKPT \
        --input_data data/mbpp/new_mbpp.json \
        --save_dir $OUTPUT_DIR/mbpp_chat \
        --num-samples-per-task $NUM_SAMPLES \

fi


if [[ " ${my_array[@]} " =~ " leetcode " ]]; then
    conda activate prime
    echo "running leetcode"
    mkdir -p $OUTPUT_DIR/leetcode_chat
    python3 Coding/leetcode/evaluate_leetcode.py \
        --model $MODEL_CKPT \
        --input_data data/leetcode/leetcode-test.json \
        --save_dir $OUTPUT_DIR/leetcode_chat \
        --num-samples-per-task $NUM_SAMPLES \

fi


if [[ " ${my_array[@]} " =~ " amc " ]]; then
    conda activate prime
    echo "running amc_chat(numina)"
    mkdir -p $OUTPUT_DIR/amc_chat
    python3 -u Math/amc/evaluate_amc.py \
        --model $MODEL_CKPT \
        --data_dir  data/AI-MO/aimo-validation-amc \
        --save_dir $OUTPUT_DIR/amc_chat \
        --num-samples-per-task $NUM_SAMPLES \

fi

if [[ " ${my_array[@]} " =~ " aime " ]]; then
    conda activate prime
    # AIME2024 chat
    echo "running aime_chat(numina)"
    mkdir -p $OUTPUT_DIR/aime_chat
    python3 -u Math/aime/evaluate_aime.py \
        --model $MODEL_CKPT \
        --data_dir  data/AI-MO/aimo-validation-aime \
        --save_dir $OUTPUT_DIR/aime_chat \
        --num-samples-per-task $NUM_SAMPLES \

fi


if [[ " ${my_array[@]} " =~ " math500 " ]]; then
    conda activate prime
    # math chat
    echo "running math_chat 500"
    mkdir -p $OUTPUT_DIR/math_chat
    python3 -u Math/math/evaluate_math.py \
        --model $MODEL_CKPT \
        --data_dir data/math500 \
        --save_dir $OUTPUT_DIR/math_chat \
        --num-samples-per-task $NUM_SAMPLES \
        
fi

if [[ " ${my_array[@]} " =~ " qwen " ]]; then
    conda activate qwen_math
    echo "running qwen math eval datasets"
    cd ./Math/Qwen25-Math/evaluation
    PROMPT_TYPE="qwen25-math-cot"
    MODEL_NAME_OR_PATH=$MODEL_CKPT
    mkdir -p $OUTPUT_DIR/qwen_math
    bash sh/eval.sh $PROMPT_TYPE $MODEL_NAME_OR_PATH $OUTPUT_DIR/qwen_math
    cd ../../../
fi


if [[ " ${my_array[@]} " =~ " livecodebench " ]]; then
    conda activate lcb
    echo "running livecodebench"
    cd ./Coding/livecodebench/LiveCodeBench-main
    mkdir -p $OUTPUT_DIR/livecodebench
    python -m lcb_runner.runner.main --model $MODEL_CKPT --scenario codegeneration --evaluate --release_version release_v4 --output_path $OUTPUT_DIR/livecodebench
    # v2
    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-05-31 >$OUTPUT_DIR/livecodebench/lcb_v2.txt 2>&1 &
    # v3
    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-08-03 >$OUTPUT_DIR/livecodebench/lcb_v3.txt 2>&1 &
    # v4
    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2023-05-01 --end_date 2024-11-01 >$OUTPUT_DIR/livecodebench/lcb_v4.txt 2>&1 &
    # 08-
    nohup python -m lcb_runner.evaluation.compute_scores --eval_all_file $OUTPUT_DIR/livecodebench/result_eval_all.json --start_date 2024-08-01 --end_date 2024-11-01 >$OUTPUT_DIR/livecodebench/lcb_08_11.txt 2>&1 &
    cd ../../../
fi

echo "Evaluation completed. Results saved to: $OUTPUT_DIR"
echo "Pass@k results are available in pass_at_k.json files within each subdirectory."
echo "To plot pass@k curves, use: python utils/plot_pass_at_k.py --results_file <path_to_pass_at_k.json> --output_dir <plot_dir>"

# Usage examples:
# ./run.sh /path/to/model                    # Use default: 10 samples, temperature 0.8
# ./run.sh /path/to/model 20                 # Use 20 samples, temperature 0.8
# ./run.sh /path/to/model 10 0.0             # Use 10 samples, temperature 0.0 (deterministic)
# ./run.sh /path/to/model 5 1.0              # Use 5 samples, temperature 1.0 (high diversity)
