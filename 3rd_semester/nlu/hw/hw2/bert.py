import os
import logging
from tqdm import tqdm, trange

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import (BertPreTrainedModel, BertTokenizer, BertModel, BertConfig,
                          AdamW, get_linear_schedule_with_warmup)
from utils import init_logger, set_seed
from data import load_examples

from torchcrf import CRF
from tomark import Tomark
from torch.utils.tensorboard import SummaryWriter

from utils import get_intent_labels, get_slot_labels, read_input_file
from eval import conlleval


logger = logging.getLogger(__name__)
tb_logger = SummaryWriter()


class IntentClassifier(nn.Module):
    def __init__(self, input_dim, num_intent_labels, dropout_rate=0.):
        super(IntentClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_intent_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class SlotClassifier(nn.Module):
    def __init__(self, input_dim, num_slot_labels, dropout_rate=0.):
        super(SlotClassifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_slot_labels)

    def forward(self, x):
        x = self.dropout(x)
        return self.linear(x)


class JointBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, slot_label_lst):
        super(JointBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits


class BertTrainer(object):
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config = BertConfig.from_pretrained(args.model_name_or_path)
        self.model = JointBERT.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst)

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        for epoch in range(int(self.args.num_train_epochs)):
            for step, batch in enumerate(pbar := tqdm(train_dataloader)):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4]}
                inputs['token_type_ids'] = batch[2]
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                pbar.set_description(f"Epoch={epoch+1}, loss={loss.item():.4f}")
                if global_step % self.args.logging_steps == 0:
                    tb_logger.add_scalars('loss', {'train':tr_loss / global_step}, global_step)
                    self.evaluate(global_step)

                if 0 < self.args.max_steps < global_step:
                    break

            if 0 < self.args.max_steps < global_step:
                break

        return global_step, tr_loss / global_step

    def evaluate(self, global_step):
        dev_sampler = RandomSampler(self.dev_dataset)
        dev_dataloader = DataLoader(self.train_dataset, sampler=dev_sampler, batch_size=self.args.eval_batch_size)
        tr_loss = 0.0
        dev_global_step = 0
        for _, batch in enumerate(tqdm(dev_dataloader, desc="Evaluating")):
            dev_global_step += 1
            batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'intent_label_ids': batch[3],
                      'slot_labels_ids': batch[4]}
            inputs['token_type_ids'] = batch[2]
            outputs = self.model(**inputs)
            loss = outputs[0]
            tr_loss += loss.item()
        tb_logger.add_scalars('loss', {"validation": tr_loss / dev_global_step}, global_step)


    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = JointBERT.from_pretrained(self.args.model_dir,
                                                   args=self.args,
                                                   intent_label_lst=self.intent_label_lst,
                                                   slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")


def evaluate(args, tokenizer, dataset):
    # load model and args
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model = JointBERT.from_pretrained(args.model_dir,
                                      args=args,
                                      intent_label_lst=get_intent_labels(args),
                                      slot_label_lst=get_slot_labels(args))
    model.to(device)
    model.eval()
    logger.info(args)

    slot_label_lst = get_slot_labels(args)
    intent_label_lst = get_intent_labels(args)

    # Convert input file to TensorDataset
    pad_token_label_id = args.ignore_index
    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)

    all_slot_label_mask = None
    intent_preds = None
    slot_preds = None

    sentences = list()
    ground_slot_labels = list()
    ground_intent_labels = list()
    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                      "attention_mask": batch[1],
                      "intent_label_ids": None,
                      "slot_labels_ids": None}

            sentences.extend(tokenizer.batch_decode(batch[0].detach().cpu().numpy(), skip_special_tokens=True))
            ground_slot_labels.extend(tokenizer.batch_decode(batch[4].detach().cpu().numpy(), skip_special_tokens=True))
            inputs["token_type_ids"] = batch[2]
            outputs = model(**inputs)
            _, (intent_logits, slot_logits) = outputs[:2]

            # Intent Prediction
            if intent_preds is None:
                ground_intent_labels = batch[3].detach().cpu().numpy()
                intent_preds = intent_logits.detach().cpu().numpy()
            else:
                ground_intent_labels = np.append(ground_intent_labels, batch[3].detach().cpu().numpy())
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()
                all_slot_label_mask = batch[4].detach().cpu().numpy()
            else:
                if args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                all_slot_label_mask = np.append(all_slot_label_mask, batch[4].detach().cpu().numpy(), axis=0)

    intent_preds = np.argmax(intent_preds, axis=1)

    if not args.use_crf:
        slot_preds = np.argmax(slot_preds, axis=2)

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    intent_label_map = {i: label for i, label in enumerate(intent_label_lst)}
    slot_preds_list = [[] for _ in range(slot_preds.shape[0])]
    slots_ground_list = [[] for _ in range(slot_preds.shape[0])]
    words = [sent.split() for sent in sentences]

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
                slots_ground_list[i].append(slot_label_map[all_slot_label_mask[i][j]])

    stats = conlleval(slot_preds_list, slots_ground_list, words, 'evaluate.txt')
    results = [
        {
            'Slot/recall': stats['r'],
            'Slot/precision': stats['p'],
            'Slot/F1': stats['f1'],
            'Intent/accuracy': np.round((intent_preds == ground_intent_labels).mean()*100, 2),
        }
    ]
    logger.info("***** Results *****")
    markdown = Tomark.table(results)
    tb_logger.add_text('result/bert', markdown)
    print(markdown)

    random_samples = np.random.randint(0, len(slot_preds_list), size=10)
    samples = list()
    for sint in random_samples:
        samples.append({
            "Sentence": ' '.join(words[sint]),
            "Slot/Ground truth": ' '.join(slots_ground_list[sint]),
            "Slot/Prediction": ' '.join(slot_preds_list[sint]),
            "Intent/Ground truth": intent_label_map[ground_intent_labels[sint]],
            "Intent/Prediction": intent_label_map[intent_preds[sint]],
        })
    markdown = Tomark.table(samples)
    tb_logger.add_text('samples/bert', markdown)
    print(markdown)


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    train_dataset = load_examples(args, tokenizer, mode="train")
    dev_dataset = load_examples(args, tokenizer, mode="dev")
    test_dataset = load_examples(args, tokenizer, mode="test")

    bert_trainer = BertTrainer(args, train_dataset, dev_dataset, test_dataset)
    bert_trainer.train()
    evaluate(args, tokenizer, test_dataset)
