from collections import namedtuple

import yaml

import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import GPT2Tokenizer


tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
tokenizer.pad_token = tokenizer.eos_token

DefaultConfig = namedtuple("DefaultConfig",
                           "epochs lr schedule_type model_version")


def batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def extract_problem(example):
    if len(example['description'].split("\n\n\nConstraints\n", 1)) > 1:
        problem = example['description'].split("\n\n\nConstraints\n", 1)[0]
    else:
        problem = example['description'].split("\n\n\nExample\n", 1)[0]
    return {"problem": problem}


def extract_solution(example):
    example["solutions.solution"] = example["solutions.solution"][0]
    return example


def tokenize_train_data(example):
    """

    :return:
    """
    return tokenizer(
        example["input_text"],
        max_length=450,
        padding='max_length',
        truncation=True
    )


def tokenize_test_data(example):
    """

    :return:
    """
    return tokenizer(
        example["solutions.solution"]+"$$:",
        text_target=example['problem'],
        max_length=450,
        padding='max_length',
        truncation=True
    )


def get_raw_dataset(split=None):
    """
    load dataset
    flatten
    filter
    extract problem and solution
    tokenize
    convert to tensor
    remove unwanted columns
    :return:
    """
    if split:
        dataset = load_dataset("deepmind/code_contests", split=split)
    else:
        dataset = load_dataset("deepmind/code_contests")
    dataset = dataset.flatten()
    dataset = dataset.filter(lambda example: len(example['solutions.solution']) > 0)
    dataset = dataset.map(extract_problem, batched=True)
    dataset = dataset.map(extract_solution, batched=True)
    dataset = dataset.map(tokenize_test_data, batched=True, remove_columns=dataset["train"].column_names)
    dataset.set_format("torch")
    return dataset


def load_artifact_dataset(wandb_run,
                          artifact="code-contests",
                          version="v0",
                          dir_name='processed_data',
                          split=None):
    dataset_artifact = wandb_run.use_artifact(f"{artifact}:{version}")
    dataset_artifact.download()
    if split:
        dataset = load_from_disk(f'artifacts/{artifact}:{version}/{dir_name}/{split}')
        dataset = dataset.map(tokenize_test_data, remove_columns=dataset.column_names)
    else:
        dataset = load_from_disk(f'artifacts/{artifact}:{version}/{dir_name}')
        dataset = dataset.map(tokenize_train_data, batched=True, remove_columns=dataset["train"].column_names)
    dataset.set_format("torch")
    return dataset


def post_process(predictions, labels):
    """

    :return:
    """
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels


def get_config_data():
    """

    :return:
    """
    config_path = "config.yml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config
