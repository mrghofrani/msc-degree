import os
import json
import random
import argparse

import torch

import numpy as np
import pandas as pd

import evaluate
from datasets import Dataset, DatasetDict
from transformers.integrations import TensorBoardCallback
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer, AutoConfig


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# torch.use_deterministic_algorithms(True)


def prepare_train_features(examples, args=None, tokenizer=None):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        max_length=args.max_length,
        stride=args.doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,)

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []
    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != 1:
                token_start_index += 1
            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != 1:
                token_end_index -= 1
            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples


def load_dataset(args):
    train = pd.read_csv(args.train_file)
    train['answers'] = train['answers'].apply(json.loads)
    validation = pd.read_csv(args.validation_file)
    validation['answers'] = validation['answers'].apply(json.loads)

    dataset = DatasetDict({
        'train': Dataset.from_pandas(train, split="train"),
        'validation': Dataset.from_pandas(validation, split="validation"),
    })

    return dataset


def compute_metrics(eval_preds):
    logits, labels = eval_preds

    predictions = np.argmax(logits, axis=-1)
    predictions = predictions.flatten()
    labels_ = np.array(labels)
    labels_ = labels_.flatten()

    acc = evaluate.load("accuracy")
    recall = evaluate.load("recall")
    precision = evaluate.load("precision")
    f1 = evaluate.load("f1")

    return {
        'accuracy': acc.compute(predictions=predictions, references=labels_)['accuracy'],
        'recall': recall.compute(predictions=predictions, references=labels_, average='macro')['recall'],
        'precision': precision.compute(predictions=predictions, references=labels_, average='macro')['precision'],
        'f1': f1.compute(predictions=predictions, references=labels_, average='macro')['f1']
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_model", required=True, type=str, help="The name of pretrained model from huggingface.")
    parser.add_argument("--train_file", required=True, type=str, help="The input training file in csv.")
    parser.add_argument("--validation_file", required=True, type=str, help="The input evaluation file in csv.")
    parser.add_argument('--experiment_name', required=True, type=str, help="The name of the experiment going to be done.")

    parser.add_argument("--use_pretrain_weights", default=True, action=argparse.BooleanOptionalAction, help="Whether to use pretrained weights.")
    parser.add_argument('--max_length', default=512, type=int, help="The maximum length of combination of question and context.")
    parser.add_argument('--doc_stride', default=256, type=int, help="The authorized overlap between two part of the context when splitting it is needed.")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size for training.")
    parser.add_argument("--evaluation_batch_size", default=16, type=int, help="Batch size for evaluation.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=5, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--weight_decay", default=0.001, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--fp16_training", default=True, action=argparse.BooleanOptionalAction, help="Whether using fp16 training mode or not.")
    parser.add_argument("--evaluation_steps", default=200, type=int, help="Number of steps to evaluate model")
    parser.add_argument("--saving_steps", default=2000, type=int, help="Number of steps to save model")
    args = parser.parse_args()

    if os.path.exists(args.experiment_name):
        assert False, "Experiment name already exists."

    dataset = load_dataset(args)

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
    tokenized_ds = dataset.map(prepare_train_features, batched=True, fn_kwargs={"tokenizer": tokenizer, "args": args},
                               remove_columns=dataset["train"].column_names)

    if args.use_pretrain_weights:
        model = AutoModelForQuestionAnswering.from_pretrained(args.pretrain_model)
    else:
        config = AutoConfig.from_pretrained(args.pretrain_model)
        model = AutoModelForQuestionAnswering.from_config(config)

    trainer_args = TrainingArguments(
        args.experiment_name,
        do_train=True,
        do_eval=True,
        evaluation_strategy = "steps",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.evaluation_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.saving_steps,
        eval_steps=args.evaluation_steps,
        fp16=args.fp16_training,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay)


    trainer = Trainer(
        model=model,
        args=trainer_args,
        compute_metrics=compute_metrics,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['validation'],
        tokenizer=tokenizer,
        callbacks=[TensorBoardCallback])

    # start training
    trainer.train()
    trainer.save_model(args.experiment_name)


if __name__ == '__main__':
    main()
