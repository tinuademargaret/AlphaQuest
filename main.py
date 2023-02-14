import os

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

config = get_config_data()

dataset = load_artifact_dataset()
num_test_solutions = int(config["num_test_solutions"])
test_solutions = dataset["test"][:num_test_solutions]

model = GPT2LMHeadModel.from_pretrained(config["model_type"])
model = model.to(device)
model.resize_token_embeddings(len(tokenizer))
model_path = os.path.join(os.getcwd(), config["model_path"])

num_epochs = int(config["num_epochs"])
learning_rate = float(config["learning_rate"])
optimizer = AdamW(model.parameters(), lr=learning_rate)
log_interval = int(config["log_interval"])
batch_size = int(config["batch_size"])

wandb.login()
wandb_config = {
      "log_interval": log_interval,
      "epochs": num_epochs
      }
wandb.init(
      project="AlphaQuest",
      config=wandb_config
      )

alpha_quest_model = AlphaQuestModel(dataset,
                                    model,
                                    model_path,
                                    device,
                                    tokenizer,
                                    batch_size
                                    )

if __name__ == '__main__':
    alpha_quest_model.train(num_epochs,
                            optimizer,
                            wandb,
                            log_interval
                            )
    eval_scores = alpha_quest_model.eval()
    alpha_quest_model.generate_problems(test_solutions)

