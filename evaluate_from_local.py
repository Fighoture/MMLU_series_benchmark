import csv
import json
import argparse
import os
import torch
import numpy as np
import pandas as pd
import random
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, AutoConfig, StoppingCriteria
import transformers
import time
import re
from vllm import LLM, SamplingParams
from tqdm import tqdm
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from distutils.util import strtobool
import logging
import sys
from datasets import load_dataset

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
answer_to_idx = {'A':0, 'B':1, 'C':2, 'D':3}
mmlu_subject_to_pruned_mask = {"college_mathematics": "MathInstruct",
                               "college_computer_science": "code_alpaca_20k",
                               "high_school_mathematics": "MathInstruct",
                               "high_school_computer_science": "code_alpaca_20k",
                               "high_school_microeconomics": "finance_alpaca"}
mmlupro_subject_to_pruned_mask = {"computer_science": "code_alpaca_20k",
                                  "business": "finance_alpaca",
                                  "math": "MathInstruct"}
max_model_length = 4096
max_new_tokens = 256

class StopOnKeyword(StoppingCriteria):
    def __init__(self, tokenizer, stop_string, existing_number=1):
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.existing_number = existing_number

    def __call__(self, input_ids, scores, **kwargs):
        for i in range(input_ids.shape[0]):
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            stop_string_occurrences = len(text.split(self.stop_string)) - 1
            if stop_string_occurrences <= self.existing_number:
                return False
        return True

def load_mmlu():
    mmlu_dataset_path = "mmlu_dataset/data"
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(mmlu_dataset_path, "test")) if "_test.csv" in f])
    val_result = []
    test_result = []
    for subject in subjects:
        val_df = pd.read_csv(os.path.join(mmlu_dataset_path, "val", subject + "_val.csv"), header=None)
        test_df = pd.read_csv(os.path.join(mmlu_dataset_path, "test", subject + "_test.csv"), header=None)
        mmlu_preprocess(val_df, subject, val_result)
        mmlu_preprocess(test_df, subject, test_result)
    return test_result, val_result

def load_mmlu_pro():
    mmlupro_dataset_path = "mmlupro_dataset"
    dataset = load_dataset(mmlupro_dataset_path)
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = mmlupro_preprocess(test_df)
    val_df = mmlupro_preprocess(val_df)
    return test_df, val_df


def mmlu_preprocess(df, subject, result):
    for i in range(df.shape[0]):
        data = df.loc[i].tolist()
        new_res = {}
        qid = len(result)
        new_res["question_id"] = qid
        new_res["question"] = data[0]
        new_res["options"] = data[1:5]
        new_res["answer"] = data[5]
        new_res["answer_index"] = answer_to_idx[data[5]]
        new_res["category"] = subject
        result.append(new_res)


def mmlupro_preprocess(test_df):
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df


def load_peft_model(model, peft_model_path):
    model = PeftModel.from_pretrained(model, peft_model_path)
    model = model.merge_and_unload()
    return model


def load_model(prune_subject=None):
    if args.mode == "vllm":
        print("vllm model loading")
        llm = LLM(model=args.model, dtype=torch.bfloat16, gpu_memory_utilization=float(args.gpu_util),
                  tensor_parallel_size=len(args.cuda_visible_device.split(",")), max_model_len=max_model_length)
        sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens,
                                         stop=["Question:"])
        print("vllm model loading finish.")
    elif args.mode == "transformers":
        print("transformers model loading")
        if prune_subject is not None:
            assert args.prune_result != "."
            config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            config.pre_ffn_hidden = True
            if args.selected_dataset == "mmlu":
                dataset_name = mmlu_subject_to_pruned_mask[prune_subject]
            else:
                dataset_name = mmlupro_subject_to_pruned_mask[prune_subject]

            if args.prune_result.endswith(".json"):
                pruning_file_path = args.prune_result
            elif f"{dataset_name}_prune.json" in os.listdir(args.prune_result):
                pruning_file_path = f"{args.prune_result}/{dataset_name}_prune.json"
            elif "c4_prune.json" in os.listdir(args.prune_result):
                pruning_file_path = f"{args.prune_result}/c4_prune.json"
            else:
                raise FileNotFoundError("Could not find pruning file.")
            print(f"pruning_file_path: {pruning_file_path}")
            with open(pruning_file_path, "r") as f:
                pruned_mask = json.load(f)
            config.pruned_mask = pruned_mask
            config.max_position_embeddings = max_model_length

            llm = AutoModelForCausalLM.from_pretrained(
                args.model,
                config=config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            llm = AutoModelForCausalLM.from_pretrained(
                args.model,
                device_map="auto",
                max_position_embeddings=max_model_length,
                trust_remote_code=True,
            )
        llm.generation_config = GenerationConfig.from_pretrained(args.model)
        llm.generation_config.pad_token_id = llm.generation_config.eos_token_id
        if args.peft_model != ".":
            dir_list = os.listdir(args.peft_model)
            if "adapter_model.safetensors" not in dir_list:
                for dir_path in dir_list:
                    peft_model_path = os.path.join(args.peft_model, dir_path)
                    llm = load_peft_model(llm, peft_model_path)
                    print(f"model peft merge from {peft_model_path}")
            else:
                llm = load_peft_model(llm, args.peft_model)
                print(f"model peft merge from {args.peft_model}")

        print(f"transformers model loading finish. Subject is {prune_subject}")
        sampling_params = None
    else:
        ValueError("Mode value error!")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print("tokenizer loaded.")
    return (llm, sampling_params), tokenizer


def args_generate_path(input_args):
    scoring_method = "CoT"
    model_name = input_args.model.split("/")[-1]
    if "/result/" in input_args.peft_model:
        save_pre_list = input_args.peft_model.split("/result/")[-1].split("/")
    elif "/pruned_result/" in input_args.prune_result:
        save_pre_str = input_args.prune_result.split("/pruned_result/")[-1]
        if save_pre_str.endswith(".json"):
            flag = save_pre_str.split("/")[-1].split(".json")[0]
            save_pre_list = save_pre_str.split("/")[:-1]
            save_pre_list = ["pruned"] + save_pre_list + [flag]
        else:
            save_pre_list = save_pre_str.split("/")
            save_pre_list = ["pruned"] + save_pre_list
    else:
        save_pre_list = []

    subjects = input_args.selected_subjects.replace(",", "-").replace(" ", "_")
    return [model_name] + save_pre_list + [scoring_method, subjects]


def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject or each["category"].replace(",", "-").replace(" ", "_") == subject:
            res.append(each)
    return res


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        if "cot_content" in example:
            cot_content = example["cot_content"].replace("A: Let's think step by step.",
                                                         "Answer: Let's think step by step.")
        else:
            cot_content = "Answer:\nThe answer is ({})".format(example["answer"])
        prompt += (cot_content + "\n\n")
    else:
        prompt += "Answer:"
    return prompt


def generate_cot_prompt(val_df, curr, k):
    subject = curr["category"]
    with open(f"cot_prompt_lib/initial_prompt.txt", "r") as fi:
        for line in fi.readlines():
            prompt = line
            break
        if "_" in subject:
            subject = " ".join(subject.split("_"))
        prompt = prompt.replace("{$}", subject) + "\n"

    if k > 0:
        val_df = val_df[: k]
        prompt += "Below are examples of the format to follow:\n"
        for example in val_df:
            prompt += format_cot_example(example, including_answer=True)
        prompt += "Now answer this question:\n"
    prompt += format_cot_example(curr, including_answer=False)
    return prompt


def check_exist(res, q_id):
    for each in res:
        if q_id == each["question_id"]:
            if "pred" in each:
                # logging.debug("exist, skip it")
                return True
            else:
                logging.debug("no result in exist result error")
                return False
        else:
            continue
    return False


def extract_answer(text, question):
    if question not in text:
        return None
    text = text.split(question)[1]
    pattern = r"[aA]nswer is \(?([ABCDEFGHIJ])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        logging.info("1st answer extract failed\n" + text)
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return None


def batch_inference(llm, sampling_params, inference_batch, tokenizer, stop_criteria=None):
    start = time.time()
    if sampling_params is not None:
        outputs = llm.generate(inference_batch, sampling_params)
    else:
        inference_batch_tokenized = tokenizer(inference_batch, return_tensors="pt", padding=True)
        # inference_batch = {key: value.cuda() for key, value in inference_batch.items()}
        with torch.no_grad():
            outputs_tokenized = llm.generate(**inference_batch_tokenized, max_new_tokens=max_new_tokens, stopping_criteria=[stop_criteria])
            outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)
    logging.info(str(len(inference_batch)) + " size batch costing time: " + str(time.time() - start))

    assert len(outputs) == len(inference_batch)
    response_batch = []
    for i, output in enumerate(outputs):
        if sampling_params is not None:
            generated_text = output.outputs[0].text
        else:
            generated_text = output
        response_batch.append(generated_text)
    return response_batch


def save_res(res, output_path):
    accu, corr, wrong = 0.0, 0.0, 0.0
    with open(output_path, "w") as fo:
        fo.write(json.dumps(res))
    for each in res:
        if not each["pred"]:
            random.seed(12345)
            x = random.randint(0, len(each["options"]) - 1)
            if x == each["answer_index"]:
                corr += 1
                # print("random hit.")
            else:
                wrong += 1
        elif each["pred"] == each["answer"]:
            corr += 1
        else:
            wrong += 1
    if corr + wrong == 0:
        return 0.0, 0.0, 0.0
    accu = corr / (corr + wrong)
    return accu, corr, wrong


@torch.no_grad()
def eval_cot(subject, model, tokenizer, val_df, test_df, output_path, exists_result=None):
    llm, sampling_params = model
    if not exists_result:
        res = []
    else:
        res = exists_result
    print("load exists result length", len(res))
    global choices
    logging.info("evaluating " + subject)
    batch_size = args.batch_size
    inference_batches = []
    in_batch_index = []

    for i in tqdm(range(len(test_df))):
        k = args.ntrain
        options_num = len(test_df[i]["options"])
        # if options_num != 10 and options_num != 4:
        #     print("options_num", options_num)
        curr = test_df[i]
        q_id = curr["question_id"]
        if check_exist(res, q_id):
            continue
        prompt_length_ok = False
        prompt = None

        while k >= 0 and not prompt_length_ok:
            prompt = generate_cot_prompt(val_df, curr, k)
            inputs = tokenizer(prompt, return_tensors="pt")
            # inputs = {key: value.cuda() for key, value in inputs.items()}
            length = len(inputs["input_ids"][0])
            if length < max_model_length - max_new_tokens:
                prompt_length_ok = True
            k -= 1

        inference_batches.append(prompt)
        in_batch_index.append(i)

    print(f"data loaded. CUDA: {torch.cuda.memory_allocated() / 1e9} GB")

    stop_criteria = StopOnKeyword(tokenizer=tokenizer, stop_string="Question:", existing_number=1 + args.ntrain)
    i = 0
    while i < len(test_df):
        if i + batch_size < len(test_df):
            end_index = i + batch_size
        else:
            end_index = len(test_df)

        curr_batch = inference_batches[i: end_index]
        print(f"batch start: {i}, batch end: {end_index}, batch length: {len(curr_batch)}, total batch length: {len(inference_batches)}")
        if len(curr_batch) == 0:
            break
        response_batch = batch_inference(llm, sampling_params, curr_batch, tokenizer, stop_criteria)
        index_list = in_batch_index[i: end_index]
        for j, index in enumerate(index_list):
            model_output = response_batch[j]
            curr = test_df[index]
            pred = extract_answer(model_output, curr["question"])
            curr["pred"] = pred
            curr["model_outputs"] = model_output
            res.append(curr)
        accu, corr, wrong = save_res(res, output_path)
        logging.info("this batch accu is: {}, corr: {}, wrong: {}\n".format(str(accu), str(corr), str(wrong)))
        i += batch_size
    accu, corr, wrong = save_res(res, output_path)
    return accu, corr, wrong


def main():
    if not os.path.exists(save_result_dir):
        os.makedirs(save_result_dir)
    if args.selected_dataset == "mmlupro":
        logging.info("loading mmlu-pro")
        full_test_df, full_val_df = load_mmlu_pro()
    elif args.selected_dataset == "mmlu":
        logging.info("loading mmlu")
        full_test_df, full_val_df = load_mmlu()
    else:
        raise ValueError("Selected dataset error.")
    all_subjects = []
    for each in full_test_df:
        if each["category"] not in all_subjects:
            all_subjects.append(each["category"])
    if args.selected_subjects == "all":
        selected_subjects = all_subjects
    else:
        selected_subjects = []
        args_selected = args.selected_subjects.split(",")
        for sub in all_subjects:
            for each in args_selected:
                if each.replace(" ", "_") in sub.replace(" ", "_"):
                    selected_subjects.append(sub.replace(" ", "_"))

    if args.prune_result == ".":
        model, tokenizer = load_model()
        print(f"model loaded. CUDA: {torch.cuda.memory_allocated() / 1e9} GB")

    logging.info("selected subjects:\n" + "\n".join(selected_subjects))
    sta_dict = {}
    selected_subjects = sorted(selected_subjects)
    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------category level sta------\n")
    for subject in selected_subjects:
        if args.prune_result != ".":
            model, tokenizer = load_model(subject)
        if subject not in sta_dict:
            sta_dict[subject] = {"corr": 0.0, "wrong": 0.0, "accu": 0.0}
        test_df = select_by_category(full_test_df, subject)
        val_df = select_by_category(full_val_df, subject)
        logging.info("{} test data length: {}".format(subject, len(test_df)))
        logging.info("{} val data length: {}".format(subject, len(val_df)))
        output_path = os.path.join(save_result_dir, "{}.json".format(subject))
        if os.path.exists(output_path):
            with open(output_path, "r") as fi:
                exists_result = json.load(fi)
        else:
            exists_result = []
        acc, corr_count, wrong_count = eval_cot(subject, model, tokenizer, val_df,
                                                test_df, output_path, exists_result)
        sta_dict[subject]["corr"] = corr_count
        sta_dict[subject]["wrong"] = wrong_count
        sta_dict[subject]["accu"] = acc
        with open(os.path.join(summary_path), 'a') as f:
            f.write("Average accuracy {:.4f} - {}\n".format(sta_dict[subject]["accu"], subject))
    total_corr, total_wrong = 0.0, 0.0
    for k, v in sta_dict.items():
        total_corr += v["corr"]
        total_wrong += v["wrong"]
    total_accu = total_corr / (total_corr + total_wrong + 0.000001)
    sta_dict["total"] = {"corr": total_corr, "wrong": total_wrong, "accu": total_accu}

    with open(os.path.join(summary_path), 'a') as f:
        f.write("\n------average acc sta------\n")
        weighted_acc = total_accu
        f.write("Average accuracy: {:.4f}\n".format(weighted_acc))
    with open(global_record_file, 'a', newline='') as file:
        writer = csv.writer(file)
        record = args_generate_path(args) + [time_str, weighted_acc]
        writer.writerow(record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--selected_dataset", "-data", type=str, default="mmlu-pro")
    parser.add_argument("--selected_subjects", "-sub", type=str, default="all")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--global_record_file", "-grf", type=str,
                        default="eval_record_collection.csv")
    parser.add_argument("--gpu_util", "-gu", type=str, default="0.8")
    parser.add_argument("--batch_size", "-bs", type=int, default=64)
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--peft_model", "-pm", type=str, default=".")
    parser.add_argument("--prune_result", "-pr", type=str, default=".")
    parser.add_argument("--cuda_visible_device", "-cuda", type=str, default="0")
    parser.add_argument("--mode", "-mode", type=str, default="vllm")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_device
    os.makedirs(args.save_dir, exist_ok=True)
    global_record_file = args.global_record_file
    save_result_dir = os.path.join(
        args.save_dir, "/".join(args_generate_path(args))
    )
    print(f"save result dir: {save_result_dir}")
    file_prefix = "-".join(args_generate_path(args))
    timestamp = time.time()
    time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
    file_name = f"{file_prefix}_{time_str}_summary.txt"
    summary_path = os.path.join(args.save_dir, "summary", file_name)
    os.makedirs(os.path.join(args.save_dir, "summary"), exist_ok=True)
    os.makedirs(save_result_dir, exist_ok=True)
    save_log_dir = os.path.join(args.save_dir, "log")
    os.makedirs(save_log_dir, exist_ok=True)
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(os.path.join(save_log_dir,
                                                                   file_name.replace("_summary.txt",
                                                                                     "_logfile.log"))),
                                  logging.StreamHandler(sys.stdout)])

    main()