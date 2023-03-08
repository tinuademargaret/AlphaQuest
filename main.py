import os
import logging
import sys

import torch
import wandb
from accelerate import Accelerator
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser
)
from src.arguments import (
    ModelArguments,
    TrainingArguments
)
from src.utils import (
    load_artifact_dataset,
    Tokenizer
)
from src.model import AlphaQuestModel

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, training_args = parser.parse_args_into_dataclasses()

    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    # logging for only one process per machine
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    learning_rate = training_args.learning_rate
    train_batch_size = training_args.per_device_train_batch_size
    eval_batch_size = training_args.per_device_eval_batch_size
    num_epochs = training_args.epochs
    log_interval = training_args.log_interval
    schedule_type = training_args.lr_scheduler
    wandb_config = {
        "log_interval": log_interval,
        "epochs": num_epochs
    }
    run = wandb.init(
        project="AlphaQuest",
        config=wandb_config,
        group="FSDP"
    )

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_class = Tokenizer(tokenizer)

    # config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    model = T5ForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
        )
    # model.resize_token_embeddings(len(tokenizer))
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    model = accelerator.prepare(model)

    if training_args.do_train:
        train_dataset = load_artifact_dataset(wandb_run=run,
                                              artifact=training_args.data_name,
                                              version=training_args.data_version,
                                              dir_name=training_args.data_dir,
                                              split='train')
    else:
        train_dataset = None

    if training_args.do_eval:
        eval_dataset = load_artifact_dataset(wandb_run=run,
                                             artifact=training_args.data_name,
                                             version=training_args.data_version,
                                             dir_name=training_args.data_dir,
                                             split='test')
    else:
        eval_dataset = None

    model_inputs = eval_model_inputs = None
    with accelerator.main_process_first():
        if train_dataset:
            model_inputs = train_dataset.map(
                tokenizer_class.tokenize_data,
                batched=True,
                remove_columns=train_dataset.column_names,
            )
            model_inputs.set_format(type="torch")
        if eval_dataset:
            eval_model_inputs = eval_dataset.map(
                tokenizer_class.tokenize_data,
                batched=True,
                remove_columns=eval_dataset.column_names,
            )
            eval_model_inputs.set_format(type="torch")

    output_dir = os.path.join(os.getcwd(), model_args.output_dir)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    alpha_quest_model = AlphaQuestModel(model_inputs,
                                        eval_model_inputs,
                                        model,
                                        output_dir,
                                        device,
                                        tokenizer,
                                        train_batch_size,
                                        eval_batch_size,
                                        data_collator
                                        )
    if training_args.do_train:
        alpha_quest_model.train(num_epochs,
                                optimizer,
                                run,
                                schedule_type,
                                training_args.gradient_accumulation_steps,
                                log_interval,
                                accelerator
                                )
    if training_args.do_eval:
        scores = alpha_quest_model.eval()
        print(f"BLEU score: {scores[0]['score']:.2f}")
        print(f"ROUGE score: {scores[1]}")

    if training_args.do_predict:
        alpha_quest_model.generate_problems()

    trained_model_artifact = run.Artifact("alpha_quest", type="model")
    trained_model_artifact.add_dir(output_dir)
    run.log_artifact(trained_model_artifact)

    run.finish()


if __name__ == '__main__':
    main()
