import os
import argparse

from types import SimpleNamespace

import torch
import wandb
from transformers import (
    AdamW,
    GPT2LMHeadModel
)

from src.utils import (
    get_config_data,
    load_artifact_dataset,
    tokenizer
)
from src.model import AlphaQuestModel

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

default_config = SimpleNamespace(
    epochs=10,
    lr=3e-5,
    schedule_type='linear',
    model_version='gpt2-medium'
)


def parse_args():
    argparser = argparse.ArgumentParser(description='Process hyper-parameters')
    argparser.add_argument('--epochs', type=int, default=default_config.epochs,
                           help='number of training epochs')
    argparser.add_argument('--lr', type=float, default=default_config.lr,
                           help='learning_rate')
    argparser.add_argument('--schedule_type', type=str, default=default_config.schedule_type,
                           help='learning_rate scheduler')
    argparser.add_argument('--model_version', type=str, default=default_config.model_version,
                           help='version of model to use')
    args = argparser.parse_args()
    vars(default_config).update(vars(args))
    return


config = get_config_data()


def train(train_config):
    learning_rate = train_config.lr
    batch_size = int(config["batch_size"])
    num_epochs = train_config.epochs
    log_interval = int(config["log_interval"])
    schedule_type = train_config.schedule_type
    wandb_config = {
        "log_interval": log_interval,
        "epochs": num_epochs
    }
    run = wandb.init(
        project="AlphaQuest",
        config=wandb_config
    )

    train_dataset = load_artifact_dataset(wandb_run=run,
                                          artifact=config["artifact_name"],
                                          version=config["artifact_version"],
                                          dir_name=config["artifact_dir"])

    test_dataset = load_artifact_dataset(wandb_run=run, split="test")

    test_solutions = test_dataset[:5]

    model = GPT2LMHeadModel.from_pretrained(train_config.model_version)
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))
    model_path = os.path.join(os.getcwd(), config["model_path"])

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    alpha_quest_model = AlphaQuestModel(train_dataset,
                                        test_dataset,
                                        model,
                                        model_path,
                                        device,
                                        tokenizer,
                                        batch_size
                                        )
    alpha_quest_model.train(num_epochs,
                            optimizer,
                            run,
                            schedule_type,
                            log_interval
                            )
    scores = alpha_quest_model.eval()
    print(f"BLEU score: {scores[0]['score']:.2f}")
    print(f"ROUGE score: {scores[1]}")
    alpha_quest_model.generate_problems(test_solutions)
    run.finish()


if __name__ == '__main__':
    parse_args()
    train(default_config)
