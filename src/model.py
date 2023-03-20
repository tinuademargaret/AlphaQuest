import math
import os

from tqdm.auto import tqdm

import torch
import evaluate
from torch.utils.data import DataLoader
from transformers import get_scheduler

from src.utils import (
    post_process
)


class AlphaQuestModel:

    def __init__(self,
                 train_dataset,
                 eval_dataset,
                 model,
                 output_dir,
                 device,
                 tokenizer,
                 train_batch_size,
                 eval_batch_size,
                 eval_epoch,
                 data_collator
                 ):
        if train_dataset:
            self.train_dataloader = DataLoader(
                train_dataset, shuffle=True, batch_size=train_batch_size, collate_fn=data_collator)
        if eval_dataset:
            self.eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size, collate_fn=data_collator)
        self.model = model
        self.output_dir = output_dir
        self.device = device
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_epoch = eval_epoch

    def train(self,
              num_epochs,
              optimizer,
              wandb_run,
              schedule_type,
              gradient_accumulation_steps,
              log_interval,
              accelerator,
              num_warmup_steps=100
              ):
        """

        :return:
        """
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / gradient_accumulation_steps)
        max_train_steps = num_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            schedule_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        optimizer, self.train_dataloader, self.eval_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, self.train_dataloader, self.eval_dataloader, lr_scheduler
        )

        progress_bar = tqdm(
            range(int(max_train_steps / accelerator.num_processes)),
            disable=not accelerator.is_local_main_process
        )
        completed_steps = 0

        wandb_run.watch(self.model, log_freq=100)
        self.model.train()

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        prev_loss = 10000
        for epoch in range(num_epochs):
            for step, batch in enumerate(self.train_dataloader):
                outputs = self.model(**batch)
                train_loss = outputs.loss
                train_loss = train_loss / gradient_accumulation_steps

                accelerator.backward(train_loss)

                if step % gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                    metrics = {"train_loss": train_loss, "epoch": epoch}

                    if step % log_interval == 0:
                        wandb_run.log(metrics)

                if completed_steps >= max_train_steps:
                    break

            self.model.eval()
            bleu_score = evaluate.load("sacrebleu")
            losses = []
            for step, batch in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    val_outputs = self.model(**batch)
                    generated_tokens = self.model.generate(batch["input_ids"],
                                                           attention_mask=batch["attention_mask"],
                                                           max_new_tokens=200
                                                           )
                labels = batch["labels"]
                val_loss = val_outputs.loss
                losses.append(accelerator.gather(val_loss.repeat(self.eval_batch_size)))
                decoded_preds, decoded_labels = post_process(
                    generated_tokens, labels, self.tokenizer)
                bleu_score.add_batch(
                    predictions=decoded_preds, references=decoded_labels)

            losses = torch.cat(losses)
            losses = losses[: len(self.eval_dataloader)]
            val_loss = torch.mean(losses)
            bleu = bleu_score.compute()['score']

            val_metrics = {"val_loss": val_loss, "bleu": bleu, "epoch": epoch}
            wandb_run.log({**metrics, **val_metrics})

            # Only save when the val_loss starts increasing
            if abs(prev_loss - val_loss) <= 0.01 or epoch == num_epochs - 1:
                output_file = os.path.join(self.output_dir, f"epoch_{epoch}.pkl")
                accelerator.wait_for_everyone()
                print(f"Saving epoch {epoch}")
                model = accelerator.unwrap_model(self.model)
                state_dict = model.state_dict()
                accelerator.save(
                    state_dict,
                    output_file
                )
            prev_loss = val_loss
        accelerator.wait_for_everyone()
        self.model = accelerator.unwrap_model(self.model)

    def eval(self, not_load=True):
        bleu_score = evaluate.load("sacrebleu")
        rouge_score = evaluate.load("rouge")
        if not not_load:
            self.model.load_state_dict(torch.load(
                os.path.join(self.output_dir, f"epoch_{self.eval_epoch}.pkl")))
            self.model.eval()
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader):
                generated_tokens = self.model.generate(
                    batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    max_length=200,
                )
                labels = batch["labels"]

                decoded_preds, decoded_labels = post_process(
                    generated_tokens, labels, self.tokenizer)
                bleu_score.add_batch(
                    predictions=decoded_preds, references=decoded_labels)
                rouge_score.add_batch(
                    predictions=decoded_preds, references=decoded_labels)

            bleu_results = bleu_score.compute()
            rouge_results = rouge_score.compute()
            return bleu_results, rouge_results

    def generate_problems(self, not_load=True):
        # only load if model is not trained
        if not not_load:
            self.model.load_state_dict(torch.load(
                os.path.join(self.output_dir, f"epoch_{self.eval_epoch}.pkl")))
            self.model.eval()

        problems = []
        count = 0
        for batch in self.eval_dataloader:
            if count > 10:
                break
            batch_problem = self.model.generate(batch["input_ids"].to(self.device),
                                                attention_mask=batch[
                                                    "attention_mask"].to(self.device),
                                                do_sample=True,
                                                max_new_tokens=150,
                                                num_return_sequences=10
                                                )
            problems.append(batch_problem)
            count += 1
        with open(os.path.join(
                self.output_dir, "problems.txt"), "w") as f:
            for i, problem in enumerate(problems):
                f.write("{}: {}\n".format(i, self.tokenizer.decode(
                    problem[0], skip_special_tokens=True)))
