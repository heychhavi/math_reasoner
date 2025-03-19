#!/bin/bash
# Script to run the full AI-Math-Reasoning pipeline
# This script demonstrates how to run the complete pipeline from data preprocessing
# through training, evaluation, and inference.

set -e  # Exit on error

# Parse command line arguments
MODEL_TYPE=${1:-"qwen"}
MODEL_SIZE=${2:-"7B"}
OUTPUT_DIR=${3:-"models/pipeline-output"}
NUM_AGENTS=${4:-3}
USE_VERIFICATION=${5:-true}
USE_ENSEMBLE=${6:-true}

# Set model path based on model type and size
if [ "$MODEL_TYPE" == "qwen" ]; then
    if [ "$MODEL_SIZE" == "7B" ]; then
        MODEL_PATH="Qwen/Qwen2.5-7B"
    elif [ "$MODEL_SIZE" == "1.5B" ]; then
        MODEL_PATH="Qwen/Qwen2.5-1.5B"
    else
        MODEL_PATH="Qwen/Qwen2.5-32B"
    fi
elif [ "$MODEL_TYPE" == "deepseek" ]; then
    if [ "$MODEL_SIZE" == "7B" ]; then
        MODEL_PATH="deepseek-ai/DeepSeek-Math-7B-RL"
    else
        MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    fi
else
    MODEL_PATH=$MODEL_TYPE  # Use directly as path
fi

echo "======= AI-Math-Reasoning Pipeline ======="
echo "Model: $MODEL_TYPE ($MODEL_PATH)"
echo "Output directory: $OUTPUT_DIR"
echo "Number of agents: $NUM_AGENTS"
echo "Use verification: $USE_VERIFICATION"
echo "Use ensemble: $USE_ENSEMBLE"
echo "=========================================="

# Create output directory
mkdir -p $OUTPUT_DIR
mkdir -p $OUTPUT_DIR/logs
mkdir -p $OUTPUT_DIR/models
mkdir -p $OUTPUT_DIR/results

# Step 1: Train with GRPO
echo "Step 1: Training with GRPO..."
python -m scripts.train \
    --config configs/grpo/default_config.json \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR/models/grpo

# Step 2: Train PRM
echo "Step 2: Training PRM reranker..."
python -m scripts.train_prm \
    --config configs/prm/default_config.json \
    --model_type $MODEL_TYPE \
    --model_name_or_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR/models/prm

# Step 3: Evaluate on test dataset
echo "Step 3: Evaluating model..."
python -m scripts.eval \
    --model_path $OUTPUT_DIR/models/grpo \
    --prm_model_path $OUTPUT_DIR/models/prm \
    --num_agents $NUM_AGENTS \
    --use_verification $USE_VERIFICATION \
    --use_ensemble $USE_ENSEMBLE \
    --dataset "numina/numina-math-filtered" \
    --split "test" \
    --output_file $OUTPUT_DIR/results/evaluation_results.json

# Step 4: Run inference on example problems
echo "Step 4: Running inference on example problems..."
python -m examples.simple_inference \
    --model_type $MODEL_TYPE \
    --model_name_or_path $OUTPUT_DIR/models/grpo \
    --prm_model_path $OUTPUT_DIR/models/prm \
    --num_agents $NUM_AGENTS \
    --use_verification $USE_VERIFICATION \
    --use_ensemble $USE_ENSEMBLE \
    --problem_file examples/sample_problems.txt \
    --output_file $OUTPUT_DIR/results/inference_results.json \
    --include_reasoning true

# Step 5 (Optional): Knowledge distillation if a smaller model is needed
if [ "$MODEL_SIZE" != "1.5B" ]; then
    echo "Step 5: Running knowledge distillation to smaller model..."
    python -m scripts.train_distillation \
        --config configs/distillation/default_config.json \
        --teacher_model_path $OUTPUT_DIR/models/grpo \
        --student_model_type qwen \
        --student_model_name Qwen/Qwen2.5-1.5B \
        --output_dir $OUTPUT_DIR/models/distilled
        
    echo "Step 6: Evaluating distilled model..."
    python -m scripts.eval \
        --model_path $OUTPUT_DIR/models/distilled \
        --num_agents $NUM_AGENTS \
        --use_verification $USE_VERIFICATION \
        --use_ensemble $USE_ENSEMBLE \
        --dataset "numina/numina-math-filtered" \
        --split "test" \
        --output_file $OUTPUT_DIR/results/distilled_evaluation_results.json
fi

echo "Pipeline completed successfully!"
echo "Results saved to $OUTPUT_DIR/results/"
