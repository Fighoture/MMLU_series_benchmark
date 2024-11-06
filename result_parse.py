import json
import sys
import re

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
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


path = sys.argv[1]
with open(path, 'r') as f:
    entries = json.load(f)
    success_pred = 0
    for e in entries:
        if e["pred"] is None:
            continue

        if e["pred"] != e["answer"]:
            import pdb
            pdb.set_trace()
            print(e["model_outputs"].split(e["question"])[1])

