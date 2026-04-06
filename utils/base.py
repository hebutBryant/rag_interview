import datetime
import json
import os
import random
import re
from typing import Dict, List, Optional

import yaml


def get_base_dir():
    # co-locate depcache intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("DEPCACHE_BASE_DIR"):
        depcache_dir = os.environ.get("DEPCACHE_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        depcache_dir = os.path.join(cache_dir, "DepCacheV2")
    os.makedirs(depcache_dir, exist_ok=True)
    return depcache_dir


def read_yaml(file_path: str = "../config/config.yaml") -> Dict:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


# this code from llama-index
def print_text(text: str, color: Optional[str] = None, end: str = "\n") -> None:
    _ANSI_COLORS = {
        "red": "91",  # 31
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "magenta": "35",
        "cyan": "36",
        # "pink": "38;5;200",  # 256-color mode
        "white": "97",
    }

    all_colors = {**_ANSI_COLORS}

    if color and color in all_colors:
        ansi_code = all_colors[color]
        text = f"\033[1;3;{ansi_code}m{text}\033[0m"
    elif color:
        # fallback: italic + bold if unsupported color
        # text = f"\033[1;3m{text}\033[0m"
        text = text

    print(text, end=end, flush=True)


def get_date_now(format="%Y-%m-%d %H:%M:%S"):
    return datetime.datetime.now().strftime(format)


def exists(path):
    return os.path.exists(path)


def isfile(path):
    return os.path.isfile(path)


def create_dir(path=None):
    if path and not exists(path):
        os.makedirs(path, exist_ok=True)


def parse_num(filename, mode, type=float, num=1, start=False):
    assert exists(filename), f"{filename} not exist!"
    assert isfile(filename), f"{filename} not a file!"

    ret = []
    with open(filename) as f:
        for line in f.readlines():
            if line.find(mode) >= 0:
                if start:
                    numbers = re.findall(r"\d+\.?\d*", line[line.find(start) :])
                else:
                    numbers = re.findall(r"\d+\.?\d*", line[line.find(mode) :])
                numbers = [type(x) for x in numbers][:num]
                ret.append(numbers if len(numbers) > 1 else numbers[0])
    return ret


def parse_str(filename, start, end=None):
    assert exists(filename), f"{filename} not exist!"
    assert isfile(filename), f"{filename} not a file!"

    ret = []
    with open(filename) as f:
        for line in f.readlines():
            if line.find(start) >= 0:
                start_idx = len(start) + line.index(start)
                end_idx = -1 if not end else line.index(end)
                ret.append(line[start_idx:end_idx])
    return ret


def read_json(file_path: str):
    assert exists(file_path), f"{file_path} not exist!"
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


def save_json(file_path: str, data, indent=2, info=True):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=indent)
    if info:
        print(f"save {len(data)} items to {file_path}")


def read_jsonl(file_path: str) -> List[Dict]:
    assert exists(file_path), f"{file_path} not exist!"
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file if line.strip()]
    return instances


def save_jsonl(file_path: str, data):
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            # json.dump(data, file, ensure_ascii=False, indent=2)
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"save {len(data)} items to {file_path}")


def escape_str(value: str) -> str:
    if not value or len(value) == 0:
        return value

    patterns = {
        '"': "",
        "{": "",
        "}": "",
    }
    for pattern in patterns:
        if pattern in value:
            value = value.replace(pattern, patterns[pattern])
    if value[0] == " " or value[-1] == " ":
        value = value.strip()
    value = " ".join(value.split())
    return value


# this code taken from langchain
def extract_json_str(text: str) -> str:
    """Extract JSON string from text."""
    match = re.search(r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract json string from output: {text}")
    return match.group()


def checkanswer(prediction, ground_truth, verbose=False):
    """
    Check whether the predicted answer matches the ground truth.

    :param str prediction:
        The predicted answer. The input string will be converted to lowercase for comparison.

    :param ground_truth:
        Defaults to a list. If the input is a string, it will be manually converted into a list.
        Each element in the list represents a set of candidate answers.
        If it is a nested list, the question includes multiple answers that must all be correctly matched.

    :return:
        A list of binary labels, where 1 indicates a successful match and 0 indicates a mismatch.
    :rtype: List[int]

    :Example:

    >>> prediction = "The cat sits on the mat"
    >>> ground_truth = [["cat", "CAT"]]
    >>> checkanswer("cat", ground_truth)
    [1]

    >>> checkanswer("cat and mat", [["cat"], ["MAT", "mat"]])
    [1, 1]
    """
    # print(f"prediction: {prediction}, ground_truth: {ground_truth}")
    prediction = prediction.lower()
    if not isinstance(ground_truth, list):
        ground_truth = [ground_truth]
    labels = []
    for instance in ground_truth:
        flag = True
        if isinstance(instance, list):
            flag = False
            instance = [i.lower() for i in instance]
            for i in instance:
                if i in prediction:
                    flag = True
                    break
        else:
            instance = instance.lower()
            if instance not in prediction:
                flag = False
        labels.append(int(flag))

    if verbose:
        print_text(
            f"\nprediction: {prediction}, \nground_truth: {ground_truth}, \nlabels: {labels}\n",
            color="yellow",
        )

    return labels


def get_accuracy(labels, info=None):
    tt = 0
    for label in labels:
        if 0 not in label and 1 in label:
            tt += 1
    acc = tt / len(labels)

    if info:
        print_text(f"{info} accuracy {acc}\n", color="green")

    return acc


def generate_sample_idx(range_length, num):
    idx = list(range(range_length))
    if num >= range_length:
        return idx
    random.seed(2000)
    sampled_idx = random.sample(idx, num)
    return sampled_idx


def test_print_text():

    colors = ["red", "green", "yellow", "blue", "magenta", "cyan", "white", "black"]

    print("=== Color Test Start ===\n")
    for c in colors:
        try:
            print_text(f"{c:>10}: This is a test line.", color=c)
        except Exception:
            # skip unknown colors (your function just ignores them anyway)
            pass

    print("\n=== Color Test End ===")


if __name__ == "__main__":

    print("hello world.")

    test_print_text()
