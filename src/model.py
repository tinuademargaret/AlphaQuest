import math
import os

from tqdm.auto import tqdm

import torch
import evaluate
from torch.utils.data import DataLoader, SequentialSampler
from transformers import get_scheduler

from src.utils import (
    batch_to_device,
    mtl_post_process
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
            self.train_dataset = train_dataset
        if eval_dataset:
            self.eval_dataloader = DataLoader(eval_dataset, batch_size=eval_batch_size,
                                              collate_fn=data_collator)
        self.model = model
        self.output_dir = output_dir
        self.device = device
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.eval_epoch = eval_epoch
        self.data_collator = data_collator

    def train(self,
              num_epochs,
              optimizer,
              wandb_run,
              schedule_type,
              gradient_accumulation_steps,
              log_interval,
              accelerator,
              num_warmup_steps=100,
              train_data=None,
              shard=0
              ):
        """

        :return:
        """
        if train_data is None:
            train_data = self.train_dataset
        train_dataloader = DataLoader(
            train_data, batch_size=self.train_batch_size, collate_fn=self.data_collator)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
        max_train_steps = num_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            schedule_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )
        optimizer, train_dataloader, self.eval_dataloader, lr_scheduler = accelerator.prepare(
            optimizer, train_dataloader, self.eval_dataloader, lr_scheduler
        )

        progress_bar = tqdm(
            range(int(max_train_steps / accelerator.num_processes)),
            disable=not accelerator.is_local_main_process
        )
        completed_steps = 0

        wandb_run.watch(self.model, log_freq=100)

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        prev_loss = 10000

        for epoch in range(num_epochs):

            self.model.train()
            for step, (problem_batch, input_batch, output_batch) in enumerate(train_dataloader):
                problem_outputs = self.model(**problem_batch)
                input_outputs = self.model(**input_batch)
                output_outputs = self.model(**output_batch)
                train_loss = problem_outputs.loss + input_outputs.loss + output_outputs.loss
                train_loss = train_loss / gradient_accumulation_steps

                accelerator.backward(train_loss)

                if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
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
            for step, (problem_batch, input_batch, output_batch) in enumerate(self.eval_dataloader):
                with torch.no_grad():
                    problem_outputs = self.model(**problem_batch)
                    input_outputs = self.model(**input_batch)
                    output_outputs = self.model(**output_batch)

                val_loss = problem_outputs.loss + input_outputs.loss + output_outputs.loss
                losses.append(accelerator.gather(val_loss.repeat(self.eval_batch_size)))

            losses = torch.cat(losses)
            losses = losses[: len(self.eval_dataloader)]
            val_loss = torch.mean(losses)
            val_metrics = {"val_loss": val_loss, "epoch": epoch}
            wandb_run.log({**metrics, **val_metrics})

            # Only save when the val_loss starts increasing
            loss_diff = prev_loss - val_loss
            if 0 < loss_diff <= 0.09 or epoch == num_epochs - 1:
                output_file = os.path.join(self.output_dir, f"epoch_{shard}_{epoch}.pkl")
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

    def curriculum_training(self,
                            num_epochs,
                            optimizer,
                            run,
                            schedule_type,
                            gradient_accumulation_steps,
                            log_interval,
                            accelerator,
                            num_shards):
        for i in range(num_shards):
            train_data = self.train_dataset.shard(num_shards=num_shards, index=i, contiguous=True)
            self.train(num_epochs,
                       optimizer,
                       run,
                       schedule_type,
                       gradient_accumulation_steps,
                       log_interval,
                       accelerator,
                       train_data=train_data,
                       shard=i)

        self.train(num_epochs,
                   optimizer,
                   run,
                   schedule_type,
                   gradient_accumulation_steps,
                   log_interval,
                   accelerator,
                   train_data=self.train_dataset,
                   shard=num_shards)

    def eval(self, not_load=True):
        """
        Eval batch size is set to 1
        :param not_load:
        :return:
        """
        bleu_score = evaluate.load("sacrebleu")
        rouge_score = evaluate.load("rouge")
        generated_tokens = {}
        labels = {}
        if not not_load:
            self.model.load_state_dict(torch.load(
                os.path.join(self.output_dir, f"epoch_{self.eval_epoch}.pkl")))
            self.model.eval()
        with torch.no_grad():
            for problem_batch, input_batch, output_batch in tqdm(self.eval_dataloader):
                generated_tokens["problem"] = self.model.generate(
                    problem_batch["input_ids"].to(self.device),
                    attention_mask=problem_batch["attention_mask"].to(self.device),
                    max_length=200,
                )
                generated_tokens["input"] = self.model.generate(
                    input_batch["input_ids"].to(self.device),
                    attention_mask=input_batch["attention_mask"].to(self.device),
                    max_length=100,
                )
                generated_tokens["output"] = self.model.generate(
                    output_batch["input_ids"].to(self.device),
                    attention_mask=output_batch["attention_mask"].to(self.device),
                    max_length=100,
                )

                labels["problem"] = problem_batch["labels"]
                labels["input"] = input_batch["labels"]
                labels["output"] = output_batch["labels"]

                decoded_preds, decoded_labels = mtl_post_process(
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

        problem_list = []
        count = 0
        for problem_batch, input_batch, output_batch in self.eval_dataloader:
            batch_problem = []
            if count > 10:
                break
            problem = self.model.generate(problem_batch["input_ids"].to(self.device),
                                          attention_mask=problem_batch[
                                              "attention_mask"].to(self.device),
                                          max_new_tokens=200,
                                          )
            batch_problem.extend(problem)
            input = self.model.generate(input_batch["input_ids"].to(self.device),
                                        attention_mask=input_batch[
                                            "attention_mask"].to(self.device),
                                        max_new_tokens=200,
                                        )
            batch_problem.extend(input)
            output = self.model.generate(output_batch["input_ids"].to(self.device),
                                         attention_mask=output_batch[
                                             "attention_mask"].to(self.device),
                                         max_new_tokens=200,
                                         )
            batch_problem.extend(output)

            problem_list.append(batch_problem)
            count += 1
        with open(os.path.join(
                self.output_dir, "problems.txt"), "w") as f:
            for i, problems in enumerate(problem_list):
                for j, problem in enumerate(problems):
                    f.write("{}_{}: {}\n".format(i, j, self.tokenizer.decode(
                        problem, skip_special_tokens=True)))
