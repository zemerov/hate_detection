from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sys import stdin
import argparse
import time
import torch


def init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path_to_model",
        type=str,
        default=None,
        help="Path to pretrained RobertaForSequenceClassification."
    )

    return parser


def init_model(path_to_model: str = None) -> RobertaForSequenceClassification:
    model = RobertaForSequenceClassification.from_pretrained(path_to_model)

    return model


def init_tokenizer() -> RobertaTokenizer:
    roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    return roberta_tokenizer


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    tokenizer = init_tokenizer()
    model = init_model(args.path_to_model)

    for line in stdin:
        start = time.time()
        tokenized_line = tokenizer(line, return_tensors='pt', padding=True, truncation=True)

        prediction = model(tokenized_line)

        end = time.time()

        print("Prediction:", prediction.tolist())
        print("It was calculated on cpu for {} seconds".format(end - start))
