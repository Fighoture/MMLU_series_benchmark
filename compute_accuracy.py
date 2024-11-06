import glob
import sys
import json
import re
import os
import random

assert len(sys.argv) > 1, 'You need to pass the directory'
path = sys.argv[1]


def extract_answer(text, question, level):
    if question not in text:
        answer = text
    else:
        answer = text.split(question)[1]
    pattern = r".*[aA]nswer is \(?([A-J])\)?.*"
    match = re.search(pattern, answer)
    if match:
        return match.group(1)
    elif level == 'l1':
        return None
    else:
        return extract_again(answer, text, level)


def extract_again(answer, text, level):
    pattern = r".*[aA]nswer:\n? ?\(?([A-J])\)?.*"
    match = re.search(pattern, answer)
    if match:
        return match.group(1)
    elif level == 'l2':
        # print(text)
        # print("**********")
        return None
    else:
        return extract_final(answer)


def extract_final(answer):
    pattern = r"[A-J](?=[^A-J]*$)"
    match = re.search(pattern, answer)
    if match:
        return match.group(0)
    else:
        return None


def stat_by_level(name, level):
    print("Level {} regex".format(level) + '==' * 20)
    succ, fail, error = 0, 0, 0
    with open(name, 'r') as f:
        entries = json.load(f)
        for e in entries:
            if type(e) is not dict:
                continue
            pred = extract_answer(e['model_outputs'], e['question'], level)
            if pred is None:
                error += 1
            elif pred == e['answer']:
                succ += 1
            else:
                fail += 1

    print("success: {}, fail: {}, error:{}".format(succ, fail, error))
    if succ == fail == 0:
        print("acc: 0, error rate: 1")
    else:
        print("acc: {}, error rate: {}".format(succ / (succ + fail), error / len(entries)))


if os.path.isdir(path):
    for name in glob.glob(path + '/*'):
        subject = name.split('/')[-1].split(".")[0]
        print("subject: {}".format(subject))
        # stat_by_level(name, 'l1')
        stat_by_level(name, 'l2')
        # stat_by_level(name, 'l3')
elif os.path.isfile(path):
    name = path
    subject = name.split('/')[-1].split(".")[0]
    print("subject: {}".format(subject))
    # stat_by_level(name, 'l1')
    stat_by_level(name, 'l2')
    # stat_by_level(name, 'l3')
else:
    print("Neither dir nor file")
