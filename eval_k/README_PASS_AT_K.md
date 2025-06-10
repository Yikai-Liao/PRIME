# Pass@k Evaluation Implementation

This document describes the modifications made to support pass@k evaluation instead of pass@1.

## Overview

The pass@k metric is calculated using the formula:
```
pass@k := E[1 - C(n-c, k) / C(n, k)]
```

Where:
- n = number of samples generated per task
- c = number of correct samples among n samples
- k = number of samples to consider (k ≤ n)
- C(a, b) = binomial coefficient "a choose b"

## Modified Files

All evaluation scripts in the `eval_k` directory have been modified to support pass@k evaluation:

### 1. Human Eval Evaluation (`Coding/human_eval/evaluate_human_eval.py`)

**Changes:**
- Default `num-samples-per-task` changed from 1 to 10
- Modified `SamplingParams` to generate n samples per question
- Updated completion extraction to handle multiple samples
- Modified sample creation to generate samples for each completion
- Enhanced evaluation to calculate pass@k for k ∈ [1, min(n, 10)]
- Added JSON output for pass@k results

**Key Features:**
- Generates n samples per task using VLLM sampling
- Saves complete test records in `samples.jsonl`
- Calculates pass@k for multiple k values
- Outputs results to both `result.txt` and `pass_at_k.json`

### 2. LeetCode Evaluation (`Coding/leetcode/evaluate_leetcode.py`)

**Changes:**
- Default `num-samples-per-task` changed from 1 to 10
- Default `temperature` changed from 0.0 to 0.8
- Modified `SamplingParams` to generate n samples per question
- Updated completion extraction to handle multiple samples
- Enhanced evaluation to calculate pass@k for k ∈ [1, min(n, 10)]
- Added JSON output for pass@k results
- Maintains compatibility with difficulty-based analysis (Easy/Medium/Hard)

**Key Features:**
- Generates n samples per task using VLLM sampling
- Saves complete test records in `samples.jsonl`
- Calculates pass@k for multiple k values
- Outputs results to both `result.txt` and `pass_at_k.json`
- Preserves hardness-based evaluation metrics

### 3. MBPP Evaluation (`Coding/mbpp/evaluate_mbpp.py`)

**Changes:**
- Added `num-samples-per-task` parameter (default 10)
- Modified temperature to 0.8 for diverse sampling
- Updated completion generation for multiple samples
- Implemented pass@k calculation with proper combinatorial formula
- Enhanced result tracking per task

**Key Features:**
- Generates multiple diverse samples per task
- Tracks correctness for each sample
- Calculates pass@k metrics using the exact formula
- Maintains backward compatibility with legacy accuracy metrics

### 4. Math Dataset Evaluation (`Math/math/evaluate_math.py`)

**Changes:**
- Added `num-samples-per-task` parameter (default 10)
- Added `temperature` parameter (default 0.8)
- Modified `SamplingParams` to generate multiple samples per question
- Updated completion generation for multiple samples
- Implemented pass@k calculation with proper combinatorial formula
- Enhanced result tracking per task
- Maintains subject and level-based analysis

**Key Features:**
- Generates multiple diverse samples per task
- Tracks correctness for each sample using math equivalence checking
- Calculates pass@k metrics using the exact formula
- Maintains backward compatibility with legacy accuracy metrics
- Preserves detailed analysis by subject and difficulty level

### 5. AIME Evaluation (`Math/aime/evaluate_aime.py`)

**Changes:**
- Added `num-samples-per-task` parameter (default 10)
- Added `temperature` parameter (default 0.8)
- Modified `SamplingParams` to generate multiple samples per question
- Updated completion generation for multiple samples
- Implemented pass@k calculation with proper combinatorial formula
- Enhanced result tracking per task
- Maintains competition-based analysis (AIME 2024 vs all years)

**Key Features:**
- Generates multiple diverse samples per task
- Tracks correctness for each sample using math equivalence checking
- Calculates pass@k metrics using the exact formula
- Maintains backward compatibility with competition-specific analysis
- Preserves detailed breakdown by competition year

### 6. AMC Evaluation (`Math/amc/evaluate_amc.py`)

**Changes:**
- Added `num-samples-per-task` parameter (default 10)
- Added `temperature` parameter (default 0.8)
- Modified `SamplingParams` to generate multiple samples per question
- Updated completion generation for multiple samples
- Implemented pass@k calculation with proper combinatorial formula
- Enhanced result tracking per task
- Maintains competition-based analysis

**Key Features:**
- Generates multiple diverse samples per task
- Tracks correctness for each sample using math equivalence checking
- Calculates pass@k metrics using the exact formula
- Maintains backward compatibility with competition-specific analysis
- Preserves detailed breakdown by competition source

### 7. Pass@k Curve Plotting (`utils/plot_pass_at_k.py`)

**New utility script for:**
- Reading pass@k results from JSON files
- Generating pass@k performance curves
- Saving plots in PNG and PDF formats
- Displaying numerical results

## Usage

### Batch Evaluation with run.sh

The easiest way to run all evaluations is using the provided run.sh script:

```bash
# Basic usage with default settings (10 samples, temperature 0.8)
./run.sh /path/to/model

# Specify number of samples per task
./run.sh /path/to/model 20

# Specify both samples and temperature
./run.sh /path/to/model 10 0.0    # Deterministic sampling
./run.sh /path/to/model 5 1.0     # High diversity sampling
```

**Parameters:**
- `$1` - Model checkpoint path (required)
- `$2` - Number of samples per task (optional, default: 10)
- `$3` - Temperature for sampling (optional, default: 0.8)

### Individual Script Usage

### Human Eval
```bash
python eval_k/Coding/human_eval/evaluate_human_eval.py \
    --model MODEL_PATH \
    --save_dir RESULTS_DIR \
    --data_dir DATA_DIR \
    --num-samples-per-task 10 \
    --temperature 0.8
```

### LeetCode
```bash
python eval_k/Coding/leetcode/evaluate_leetcode.py \
    --model MODEL_PATH \
    --input_data DATA_FILE \
    --save_dir RESULTS_DIR \
    --num-samples-per-task 10 \
    --temperature 0.8
```

### MBPP
```bash
python eval_k/Coding/mbpp/evaluate_mbpp.py \
    --model MODEL_PATH \
    --input_data DATA_FILE \
    --save_dir RESULTS_DIR \
    --num-samples-per-task 10
```

### Math Dataset
```bash
python eval_k/Math/math/evaluate_math.py \
    --model MODEL_PATH \
    --data_dir DATA_DIR \
    --save_dir RESULTS_DIR \
    --num-samples-per-task 10 \
    --temperature 0.8
```

### AIME
```bash
python eval_k/Math/aime/evaluate_aime.py \
    --model MODEL_PATH \
    --data_dir DATA_DIR \
    --save_dir RESULTS_DIR \
    --num-samples-per-task 10 \
    --temperature 0.8
```

### AMC
```bash
python eval_k/Math/amc/evaluate_amc.py \
    --model MODEL_PATH \
    --data_dir DATA_DIR \
    --save_dir RESULTS_DIR \
    --num-samples-per-task 10 \
    --temperature 0.8
```

### Plotting Results
```bash
python eval_k/utils/plot_pass_at_k.py \
    --results_file RESULTS_DIR/pass_at_k.json \
    --output_dir PLOTS_DIR
```

### Evaluation Datasets

The run.sh script controls which datasets to evaluate by modifying the `my_array` variable:
```bash
my_array=(mbpp leetcode math500 amc aime qwen livecodebench)
```

To run specific evaluations, modify this array to include only desired datasets.

## Output Files

For each evaluation, the following files are generated:

1. **`samples.jsonl`** - Complete test records with all samples
2. **`samples.jsonl_results.jsonl`** - Detailed execution results
3. **`result.txt`** - Summary of results including pass@k metrics
4. **`pass_at_k.json`** - JSON format pass@k results for plotting
5. **`pass_at_k_curve.png/pdf`** - Performance curve plots (when using plot utility)

## Pass@k Calculation Details

The implementation uses the exact combinatorial formula:
- For each task with n samples and c correct samples
- Calculate pass@k = 1 - C(n-c, k) / C(n, k) for each k
- Average across all tasks to get final pass@k score

This provides a low-variance estimate of the pass@k metric as described in the Codex paper.

## Recommended Settings

- **num-samples-per-task**: 10-20 for reliable pass@k estimation
- **temperature**: 0.8 for diverse sampling (0.0 for deterministic)
- **k values**: Typically evaluate k ∈ [1, 5, 10] for comparison

## Modified Evaluation Scripts Summary

**Coding Tasks:**
1. **Human Eval** - Code generation benchmark
2. **LeetCode** - Programming contest problems with difficulty levels
3. **MBPP** - Mostly Basic Python Problems

**Math Tasks:**
4. **Math Dataset** - Competition math problems with subject/level analysis
5. **AIME** - American Invitational Mathematics Examination
6. **AMC** - American Mathematics Competitions

**All scripts now support:**
- Multiple sampling per task (default: 10 samples)
- Temperature-controlled diverse generation (default: 0.8)
- Pass@k calculation using exact combinatorial formula
- JSON output for plotting pass@k curves
- Backward compatibility with original metrics

## Notes

- The implementation maintains backward compatibility with existing pass@1 evaluations
- All test records are preserved for detailed analysis
- The plotting utility supports multiple result files for comparison
- Memory usage scales with num-samples-per-task × num-tasks
- Each evaluation script preserves its domain-specific analysis (difficulty levels, competition years, subjects, etc.)
- All scripts use the same pass@k formula: `pass@k := E[1 - C(n-c, k) / C(n, k)]`