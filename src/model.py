import os

from tqdm.auto import tqdm

import torch
import evaluate
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.utils import (
    batch_to_device,
    post_process,
)


class AlphaQuestModel:

    def __init__(self,
                 dataset,
                 model,
                 model_path,
                 device,
                 tokenizer,
                 batch_size
                 ):

        self.train_dataloader = DataLoader(
            dataset["train"], shuffle=True, batch_size=batch_size)
        self.eval_dataloader = DataLoader(dataset["valid"], batch_size=batch_size)
        self.test_dataloader = DataLoader(dataset["test"], batch_size=batch_size)
        self.model = model
        self.model_path = model_path
        self.device = device
        self.tokenizer = tokenizer

    def train(self,
              num_epochs,
              optimizer,
              wandb_run,
              log_interval):
        """

        :return:
        """
        accelerator = Accelerator()
        train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
            self.train_dataloader, self.eval_dataloader, self.model, optimizer
        )
        num_training_steps = num_epochs * len(self.train_dataloader)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
        )
        progress_bar = tqdm(range(num_training_steps))
        wandb_run.watch(model, log_freq=100)
        model.train()

        for epoch in range(num_epochs):
            for step, batch in enumerate(train_dataloader):
                # batch = batch_to_device(batch, self.device)

                outputs = model(**batch)
                train_loss = outputs.loss
                # train_loss.backward()
                accelerator.backward(train_loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

                metrics = {"train_loss": train_loss}

                if step % log_interval == 0:
                    wandb_run.log(metrics)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for step, batch in enumerate(eval_dataloader):
                    # batch = batch_to_device(batch, self.device)

                    val_outputs = model(**batch)
                    val_loss += val_outputs.loss

                # Average loss for the batch
                val_metrics = {"val_loss": val_loss / len(
                    eval_dataloader)}
                wandb_run.log({**metrics, **val_metrics})

        accelerator.wait_for_everyone()
        self.model = accelerator.unwrap_model(model)
        self.model.save_pretrained(os.path.join(
            self.model_path, "alpha_quest.pt"), save_function=accelerator.save)
        print("Training Completed")

    def eval(self):
        bleu_score = evaluate.load("sacrebleu")
        rouge_score = evaluate.load("rouge")

        self.model.eval()

        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                generated_tokens = self.model.generate(
                    batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    max_length=350,
                )
                labels = batch["labels"]

                decoded_preds, decoded_labels = post_process(
                    generated_tokens, labels)
                bleu_score.add_batch(
                    predictions=decoded_preds, references=decoded_labels)
                rouge_score.add_batch(
                    predictions=decoded_preds, references=decoded_labels)

            bleu_results = bleu_score.compute()
            rouge_results = rouge_score.compute()
            return bleu_results, rouge_results

    def generate_problems(self, solutions):

        problems = self.model.generate(solutions["input_ids"].to(self.device),
                                       attention_mask=solutions[
                                           "attention_mask"].to(self.device),
                                       max_length=350
                                       )

        with open(os.path.join(
                self.model_path, "problems.txt"), "w") as f:
            for i, problem in enumerate(problems):
                f.write("{}: {}".format(i, self.tokenizer.decode(
                    problem, skip_special_tokens=True)))
