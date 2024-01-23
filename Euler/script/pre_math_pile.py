# -*- coding:utf-8 -*-
# @FileName : pre_math_pile.py.py
# @Time : 2024/1/12 21:16
# @Author :fiv
import json
import os
from pathlib import Path

import jsonlines


def get_filepath(dir_path):
    p = Path(dir_path)
    all_path = p.glob("**/*.jsonl")
    filepaths = [str(path) for path in all_path if path.is_file()]
    return filepaths


def get_text(filepath):
    text = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            raw = json.loads(line)
            text.append({"text": raw['text']})
    return text


def write_text(text, filename):
    filepath = f"../data/{filename}"
    if os.path.exists(filepath):
        os.remove(filepath)
    with jsonlines.open(filepath, 'w') as writer:
        writer.write_all(text)


def main():
    dir_path = "../MathPile"
    filepaths = get_filepath(dir_path)

    for filepath in filepaths:
        text = get_text(filepath)
        filename = filepath.split('/')[-1]
        write_text(text, filename)


main()
