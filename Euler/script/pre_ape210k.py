# -*- coding:utf-8 -*-
# @FileName : pre_ape210k.py
# @Time : 2024/1/21 15:24
# @Author :fiv

import json
import os
from pathlib import Path

from jsonlines import jsonlines


def get_filepath(dir_path):
    p = Path(dir_path)
    all_path = p.glob("**/*.json")
    filepaths = [str(path) for path in all_path if path.is_file()]
    return filepaths


def get_text(filepath):
    text = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            raw = json.loads(line)
            answer = f"Equation is {raw['equation']}, Answer is {raw['ans']}"
            text.append({"problem": raw['original_text'], "answer": answer})
    return text


def write_text(text, filename):
    filepath = f"../data/{filename}"
    print(filepath)
    if os.path.exists(filepath):
        os.remove(filepath)
    with jsonlines.open(filepath, 'w') as writer:
        writer.write_all(text)


def main():
    dir_path = "../Ape210K"
    filepaths = get_filepath(dir_path)
    print(filepaths)
    for filepath in filepaths:
        text = get_text(filepath)
        filename = filepath.split('/')[-1]
        write_text(text, filename)


main()
