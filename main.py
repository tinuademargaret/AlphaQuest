import torch
import wandb
from transformers import (
    AdamW,
    GPT2LMHeadModel
)

from src.utils import (
    load_artifact_dataset,
    tokenizer
)
from src.model import AlphaQuestModel

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

dataset = load_artifact_dataset()
test_solutions = dataset["test"][:5]

model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
model = model.to(device)
model.resize_token_embeddings(len(tokenizer))
model_path = "/Users/tinuade/Documents/alpha_quest/outputs"

num_epochs = 5
optimizer = AdamW(model.parameters(), lr=5e-5)
log_interval = 100

wandb.login()
wandb_config = {
      "log_interval": 100,
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
                                    tokenizer
                                    )

if __name__ == '__main__':
    alpha_quest_model.train(num_epochs,
                            optimizer,
                            wandb,
                            log_interval
                            )
    eval_scores = alpha_quest_model.eval()
    alpha_quest_model.generate_problems(test_solutions)

