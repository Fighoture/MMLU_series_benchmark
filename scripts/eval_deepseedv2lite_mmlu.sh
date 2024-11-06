ntrain=5
save_dir="mmlu_eval_results/"
global_record_file="mmlupro_eval_results/eval_record_collection_0514_darth.csv"
model="/vepfs/hongcheng/moe/modelscope_cache/deepseek-ai/DeepSeek-V2-Lite"
#peft_model="."
#prune_result="."
#peft_model="/vepfs/hongcheng/moe/MoE_unsupervised_pruning/result/train_2500_valid_250/pruning_finetuning/sample_500_cluster_12/in_class_prune_0.2"
#prune_result="/vepfs/hongcheng/moe/MoE_unsupervised_pruning/pruned_result/sample_500_cluster_12/in_class_prune_0.2"
peft_model="."
prune_result="/vepfs/hongcheng/moe/MoE_unsupervised_pruning/pruned_result/sample_5000/seer_prune_0.2"
#peft_model="/vepfs/hongcheng/moe/MoE_unsupervised_pruning/result/train_5000_valid_500/baseline_finetuning"
#prune_result="."
selected_dataset="mmlu"
selected_subjects="college_mathematics,college_computer_science,high_school_mathematics,high_school_computer_science,high_school_microeconomics"
gpu_util=0.6
batch_size=8
mode="transformers"
CUDA_VISIBLE_DEVICES="0,1"

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

