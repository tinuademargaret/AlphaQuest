from collections import OrderedDict
import numpy as np
import torch
from datasets import load_dataset, load_from_disk


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
    else:
        dataset = load_from_disk(f'artifacts/{artifact}:{version}/{dir_name}')
    return dataset


def post_process(predictions, labels, tokenizer):
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


def mtl_post_process(predictions, labels, tokenizer):
    decoded_preds = []
    decoded_labels = []

    for k in zip(predictions.keys(), labels.keys()):
        decoded_pred, decoded_label = post_process(predictions[k], labels[k], tokenizer)
        decoded_preds.extend(decoded_pred)
        decoded_labels.extend(decoded_label)

    return decoded_preds, decoded_labels


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prefix = "Generate Problem: "
        self.languages = {0: "Unknown", 1: "Python2",
                          2: "C++", 3: "Python", 4: "Java"}

    def tokenize_data(self, examples):
        solutions = examples['solutions.solution']
        problems = examples['problem']
        tags = examples['cf_tags']
        languages = examples['solutions.language']

        # looping here so that tokenisation can be batched
        inputs = [self.prefix + "\n" + "Language:" + self.languages[language] + "Tag: " + str(tag) + solution
                  for language, tag, solution in zip(languages, tags, solutions)]

        model_inputs = self.tokenizer(inputs, truncation=True, padding='max_length', max_length=512)

        # encode the problems
        # T5forConditionalGeneration automatically prepares the decoder input from the labels
        labels = self.tokenizer(problems, truncation=True).input_ids

        # important: we need to replace the index of the padding tokens by -100
        # such that they are not taken into account by the CrossEntropyLoss
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)

        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs

    def tokenize_mtl_data(self, example):
        solution = example['solutions.solution']
        tag = example['cf_tags']
        language = example['solutions.language']
        tasks = OrderedDict([("Problem", example['problem']),
                             ("Input", example['input']),
                             ("Output", example['output']),
                             ])

        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}

        for task, task_output in tasks.items():
            input_sequence = "Generate" + task + "\n" + "Language: " + self.languages[language] + "\n" + "Tag: " + str(
                tag) + "\n" + "Solution: " + solution
            tokenized_input_sequence = self.tokenizer(input_sequence, truncation=True, padding='max_length',
                                                      max_length=512)

            task_sequence = task + task_output
            # using one context length since the pad tokens would be ignored, so we can have batch size
            if task == "Problem":
                tokenized_task_sequence = self.tokenizer(task_sequence, truncation=True, padding='max_length',
                                                         max_length=512).input_ids
            elif task == "Input":
                tokenized_task_sequence = self.tokenizer(task_sequence, truncation=True, padding='max_length',
                                                         max_length=200).input_ids
            else:
                tokenized_task_sequence = self.tokenizer(task_sequence, truncation=True, padding='max_length',
                                                         max_length=100).input_ids
            # replace pad tokens for labels to -100
            tokenized_task_sequence = [label if label != 0 else -100 for label in tokenized_task_sequence]

            model_inputs["input_ids"].append(tokenized_input_sequence["input_ids"])
            model_inputs["attention_mask"].append(tokenized_input_sequence["attention_mask"])
            model_inputs["labels"].append(tokenized_task_sequence)

        return model_inputs


def collate_fn(batch):
    """
    Used by the dataloader to seperate batch into tasks
    :param batch:
    :return:
    """

    max_len = 512

    batch_size = len(batch)
    keyz = ("input_ids", "attention_mask")

    problem_batch = {k: torch.LongTensor(batch_size, max_len).fill_(0) for k in keyz}
    input_batch = {k: torch.LongTensor(batch_size, max_len).fill_(0) for k in keyz}
    output_batch = {k: torch.LongTensor(batch_size, max_len).fill_(0) for k in keyz}
    problem_batch["labels"] = torch.LongTensor(batch_size, 512).fill_(0)
    input_batch["labels"] = torch.LongTensor(batch_size, 200).fill_(0)
    output_batch["labels"] = torch.LongTensor(batch_size, 100).fill_(0)

    for i, sample in enumerate(batch):
        for k, v in sample.items():
            problem_batch[k][i] = torch.LongTensor(v[0])
            input_batch[k][i] = torch.LongTensor(v[1])
            output_batch[k][i] = torch.LongTensor(v[2])

    return problem_batch, input_batch, output_batch
