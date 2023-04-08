import numpy as np
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


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.prefix = "Generate problem: "
        self.languages = {0: "Unknown", 1: "Python2",
                          2: "C++", 3: "Python", 4: "Java"}

    def tokenize_data(self, examples):
        solutions = examples['solutions.solution']
        problems = examples['problem']
        tags = examples['cf_tags']
        languages = examples['solutions.language']

        inputs = ["Language:" + self.languages[language] + "Tag: " + str(tag) + self.prefix + solution
                  for language, tag, solution in zip(languages, tags, solutions)]

        model_inputs = self.tokenizer(inputs, truncation=True, padding='max_length', max_length=500)

        # encode the problems
        labels = self.tokenizer(problems, truncation=True).input_ids

        # important: we need to replace the index of the padding tokens by -100
        # such that they are not taken into account by the CrossEntropyLoss
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)

        model_inputs["labels"] = labels_with_ignore_index

        return model_inputs
