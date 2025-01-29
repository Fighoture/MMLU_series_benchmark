#!/bin/bash
ntrain=5
save_dir="mmlu_eval_results/"
global_record_file="mmlu_eval_results/eval_record_collection.csv"
#model="/home/test/.cache/modelscope/hub/deepseek-ai/DeepSeek-V2-Lite"
model="/home/test/.cache/modelscope/hub/Qwen/Qwen1___5-MoE-A2___7B"
peft_model="."
selected_dataset="mmlu"
gpu_util=0.8
batch_size=8
mode="transformers"
CUDA_VISIBLE_DEVICES="2,3"

prune_result_list=(
    "/home/test/test/ghc/moe_prune/MoE_unsupervised_pruning/pruned_result/Qwen1___5-MoE-A2___7B/sample_1000/no_layerwise_prune/cluster_12/hierarchical_prune/rate_0.2"
)

#prune_result="."
#prune_result="/home/test/test/ghc/moe_prune/MoE_unsupervised_pruning/pruned_result/DeepSeek-V2-Lite/sample_1000/cluster_12/hierarchical_prune/rate_0.5"
#prune_result="/home/test/test/ghc/moe_prune/MoE_unsupervised_pruning/pruned_result/DeepSeek-V2-Lite/sample_1000/cluster_12/hierarchical_prune/rate_0.1/cluster_12/hierarchical_prune/rate_0.1"
#prune_result="/home/test/test/ghc/moe_prune/MoE_unsupervised_pruning/pruned_result/DeepSeek-V2-Lite/sample_1000/no_layerwise_prune/cluster_12/hierarchical_prune/rate_0.2"
#prune_result="/home/test/test/ghc/moe_prune/MoE_unsupervised_pruning/pruned_result/DeepSeek-V2-Lite/sample_1000/seer_prune/rate_0.2/c4_prune.json"
#prune_result="/home/test/test/ghc/moe_prune/MoE_unsupervised_pruning/pruned_result/Qwen1___5-MoE-A2___7B/sample_1000/cluster_16/hierarchical_prune/rate_0.1/cluster_16/hierarchical_prune/rate_0.1"
#prune_result="/home/test/test/ghc/moe_prune/MoE_unsupervised_pruning/pruned_result/Qwen1___5-MoE-A2___7B/sample_32/hsic_prune/rate_0.2/c4_prune.json"
#prune_result="/home/test/test/ghc/moe_prune/MoE_unsupervised_pruning/pruned_result/Qwen1___5-MoE-A2___7B/sample_1000/seer_prune/rate_0.2/c4_prune.json"

selected_subjects_list=(
    "college_mathematics,college_computer_science,high_school_mathematics,high_school_computer_science"
    "high_school_microeconomics,high_school_macroeconomics,econometrics"
)

for prune_result in "${prune_result_list[@]}"
do
    for selected_subjects in "${selected_subjects_list[@]}"
    do
        python evaluate_from_local.py \
               --ntrain $ntrain \
               --selected_subjects $selected_subjects \
               --selected_dataset $selected_dataset \
               --save_dir $save_dir \
               --model $model \
               --peft_model $peft_model \
               --prune_result $prune_result \
               --global_record_file $global_record_file \
               --gpu_util $gpu_util \
               --batch_size $batch_size \
               --cuda_visible_device $CUDA_VISIBLE_DEVICES \
               --mode $mode
    done
done
