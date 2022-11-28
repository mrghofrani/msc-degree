import os
import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import T5TokenizerFast, T5ForConditionalGeneration
from torch.utils.tensorboard import SummaryWriter
from tomark import Tomark
from eval import conlleval


# class Config:
#     TRAIN_BATCH_SIZE = 16
#     VALID_BATCH_SIZE = 128
#     TRAIN_EPOCHS = 1
#     LEARNING_RATE = 1e-4
#     SEED = 42
#     MAX_LEN = 128
#     device='cuda' if torch.cuda.is_available() else 'cpu'
#     data_dir = 'data'
#     model_dir = 'hello'
#     do_train = True
#     do_eval = True

logger = logging.getLogger(__name__)
tb_logger = SummaryWriter()


class T5Dataset(Dataset):
    def __init__(self, tokenizer, dir_path, mode, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(f'{dir_path}/{mode}/label') as f:
            intents = f.readlines()
        with open(f'{dir_path}/{mode}/seq_in') as f:
            word_seq = f.readlines()
        with open(f'{dir_path}/{mode}/seq_out') as f:
            slots_seq = f.readlines()
        self.read_dataset(word_seq, slots_seq, intents)

    def __len__(self):
        return len(self.dataset)

    def read_dataset(self, word_seq, slots_seq, intents):
        self.dataset = list()

        for words, slots in zip(word_seq, slots_seq):
            x = "slot detection: " + words
            y = slots
            self.dataset.append((x, y))

        for words, intents in zip(word_seq, intents):
            x = "intents: " + words
            y = intents
            self.dataset.append((x, y))

    def __getitem__(self, index):
        x, y = self.dataset[index]

        source = self.tokenizer([x], padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')
        target = self.tokenizer([y], padding='max_length', max_length=self.max_len, truncation=True, return_tensors='pt')

        return {
            'source_ids': source.input_ids.squeeze().to(dtype=torch.long),
            'source_mask': source.attention_mask.squeeze().to(dtype=torch.long),
            'target_ids': target.input_ids.squeeze().to(dtype=torch.long),
            'target_ids_y': target.attention_mask.squeeze().to(dtype=torch.long)
        }


def train(args, tokenizer, model, train_loader, dev_loader, optimizer):
    model.train()
    global_steps = 0
    for epoch in range(args.num_train_epochs):
        for _, data in enumerate(pbar := tqdm(train_loader)):
            global_steps += 1
            y = data['target_ids'].to(args.device, dtype = torch.long)
            lm_labels = y.clone().detach()
            lm_labels[lm_labels == tokenizer.pad_token_id] = -100
            ids = data['source_ids'].to(args.device, dtype = torch.long)
            mask = data['source_mask'].to(args.device, dtype = torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_description(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')
            if global_steps % args.logging_steps == 0:
                tb_logger.add_scalars('loss', {"train": loss.item()}, global_steps)
                evaluate(args, tokenizer, model, global_steps, dev_loader)


def evaluate(args, tokenizer, model, global_step, dev_loader):
    tr_loss = 0.0
    dev_global_step = 0
    for _, data in enumerate(tqdm(dev_loader, desc="Evaluating")):
        dev_global_step += 1
        y = data['target_ids'].to(args.device, dtype = torch.long)
        lm_labels = y.clone().detach()
        lm_labels[lm_labels == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(args.device, dtype = torch.long)
        mask = data['source_mask'].to(args.device, dtype = torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=lm_labels)
        tr_loss += outputs.loss.item()
    tb_logger.add_scalars('loss', {"validation": tr_loss / dev_global_step}, global_step)


def validate(tokenizer, model, device, loader):
    model.eval()
    slot_predictions = []
    slot_goals = []
    intent_predictions = []
    intent_goals = []
    sentences = []
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader, desc="Predicting")):
            y = data['target_ids'].to(device, dtype = torch.long)
            ids = data['source_ids'].to(device, dtype = torch.long)
            mask = data['source_mask'].to(device, dtype = torch.long)

            generated_ids = model.generate(input_ids = ids, attention_mask=mask, do_sample=False)
            sents = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            target = tokenizer.batch_decode(y, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            for sent, pred, actual in zip(sents, preds, target):
                sentences.append(sent.split())
                if 'slot detection' in sent:
                    slot_predictions.append(pred.split())
                    slot_goals.append(actual.split())
                else:
                    intent_predictions.append(pred.split())
                    intent_goals.append(actual.split())

        acc = 0
        for i_goal, i_pred in zip(intent_goals, intent_predictions):
            if i_goal == i_pred:
                acc += (1/len(intent_goals))

        stats = conlleval(slot_predictions, slot_goals, sentences, 'evaluate.txt')
        results = [
            {
                'Slot/recall': stats['r'],
                'Slot/precision': stats['p'],
                'Slot/F1': stats['f1'],
                'Intent/accuracy': np.round(acc * 100, 2),
            }
        ]
        print("***** Results *****")
        markdown = Tomark.table(results)
        tb_logger.add_text('result/bert', markdown)
        print(markdown)

        random_samples = np.random.randint(0, len(slot_predictions), size=10)
        samples = list()
        for sint in random_samples:
            samples.append({
                "Sentence": ' '.join(sentences[sint]),
                "Slot/Ground truth": ' '.join(slot_goals[sint]),
                "Slot/Prediction": ' '.join(slot_predictions[sint]),
                "Intent/Ground truth": ' '.join(intent_goals[sint]),
                "Intent/Prediction": ' '.join(intent_predictions[sint]),
            })
        markdown = Tomark.table(samples)
        tb_logger.add_text('samples/bert', markdown)
        print(markdown)


def save_model(args, model):
    # Save model checkpoint (Overwrite)
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.model_dir)

    # Save training arguments together with the trained model
    torch.save(args, os.path.join(args.model_dir, 'training_args.bin'))
    print("Saving model checkpoint to %s" % args.model_dir)


def load_model(args):
    # Check whether model exists
    if not os.path.exists(args.model_dir):
        raise Exception("Model doesn't exists! Train first!")
    try:
        model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
        model.to(args.device)
        return model
    except:
        raise Exception("Some model files might be missing...")


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    tokenizer = T5TokenizerFast.from_pretrained("t5-small", model_max_length=args.max_seq_len)
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = model.to(args.device)

    training_set = T5Dataset(tokenizer, args.data_dir, mode='train', max_len=args.max_seq_len)
    val_set = T5Dataset(tokenizer, args.data_dir, mode='dev', max_len=args.max_seq_len)
    test_set = T5Dataset(tokenizer, args.data_dir, mode='test', max_len=args.max_seq_len)

    training_loader = DataLoader(training_set, batch_size=args.train_batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(params =  model.parameters(), lr=args.learning_rate)
    print('Initiating Fine-Tuning for the model on our dataset')

    train(args, tokenizer, model, training_loader, val_loader, optimizer)
    save_model(args, model)

    model = load_model(args)
    validate(tokenizer, model, args.device, test_loader)


if __name__ == '__main__':
    main()
