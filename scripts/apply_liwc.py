#!/usr/bin/env python3
"""
Compute LIWC weights for each element in a JSON-lines file.

Author: Arya D. McCarthy
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import sys
from typing import (
    Collection,
    Dict,
    IO,
    List,
    Mapping,
    Pattern,
    NewType,
#    TypedDict,
    Union,
)

from nltk.tokenize import TweetTokenizer
import tqdm

from combine_expressions_for_search import make_pattern

Category = NewType("Category", str)
Lexicon = Mapping[Category, Pattern]
LexiconScore = Mapping[Category, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--liwc", type=argparse.FileType("rt"), required=True)
    parser.add_argument("--output")
    args = parser.parse_args(
        # ["--lexicon", "../work/liwc.json", "--input", "../work/ebola/data.json.gz"]
    )
    return args


@dataclass
class Liwc:
    """Manage and apply LIWC data."""

    lexicon: Lexicon

    @classmethod
    def from_file(cls, file: IO) -> Liwc:
        lexicon = cls._extract_mapping_from_stream(file)
        return Liwc(lexicon)

    @staticmethod
    def _extract_mapping_from_stream(file: IO) -> Lexicon:
        data_: Dict[str, List[str]] = json.load(file)
        data: Lexicon = {Category(k): make_pattern(v) for k, v in data_.items()}
        return data

    def compute_features(self, tokens: Collection[str]) -> LexiconScore:
        tokenized_text = " ".join(tokens).lower()
        try:
            return {
                category: Liwc._score_patterns(pattern, tokenized_text)
                for category, pattern in self.lexicon.items()
            }
        except ZeroDivisionError:  # Empty list of tokens passed. (Empty string input?)
            return {
                category: float("nan") for category, patterns in self.lexicon.items()
            }

    @staticmethod
    def _score_patterns(pattern: Pattern, text: str) -> float:
        match_count: int = len(pattern.findall(text))
        return match_count / len(text.split())


class StudyItem(Dict):
    id: str
    text: str
    supervised_annotations: Dict[str, Union[str, float]]


def main():
    args = parse_args()
    liwc = Liwc.from_file(args.liwc)

    tweet_tokenizer = TweetTokenizer()

    with open(args.schema, "rt") as ifd:
        schema = json.loads(ifd.read())

    with gzip.open(args.output, "wt") as ofd:
        with gzip.open(args.data, "rt") as reader:
            for line in tqdm.tqdm(reader):
                j: StudyItem = json.loads(line)
                texts = []
                for k, v in j.items():
                    if schema["data_fields"].get(k, {}).get("type", None) == "text":
                        texts.append(tweet_tokenizer.tokenize(v))
                tokens = sum(texts, [])
                j["liwc_features"] = liwc.compute_features(tokens)
                ofd.write(json.dumps(j) + "\n")


if __name__ == "__main__":
    main()
