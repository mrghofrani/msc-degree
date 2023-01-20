import json
import argparse

import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

import evaluate
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


class AnswerEvaluator:
    def __init__(self, args, model, tokenizer):
        self.model = model.eval().to(args.device)
        self.tokenizer = tokenizer
        self.device = args.device
        self.max_length = args.max_length
        self.stride = args.doc_stride
        self.n_best = args.n_best

    def model_pred(self, questions, contexts, batch_size=1):
        n = len(contexts)
        if n%batch_size!=0:
            raise Exception("batch_size must be divisible by sample length")

        tokens = self.tokenizer(questions, contexts, add_special_tokens=True,
                                return_token_type_ids=True, return_tensors="pt", padding=True,
                                return_offsets_mapping=True, truncation="only_second",
                                max_length=self.max_length, stride=self.stride)

        start_logits, end_logits = [], []
        for i in tqdm(range(0, n-batch_size+1, batch_size)):
            with torch.no_grad():
                out = self.model(tokens['input_ids'][i:i+batch_size].to(self.device),
                            tokens['attention_mask'][i:i+batch_size].to(self.device),
                            tokens['token_type_ids'][i:i+batch_size].to(self.device))

                start_logits.append(out.start_logits)
                end_logits.append(out.end_logits)

        _start_logits = torch.stack(start_logits).view(n, -1)
        start_logits = F.softmax(_start_logits, dim=1)

        _end_logits = torch.stack(end_logits).view(n, -1)
        end_logits = F.softmax(_end_logits, dim=1)

        return tokens, start_logits, end_logits

    def evaluate(self, args):
        df = pd.read_csv(args.test_file)
        df['answers'] = df['answers'].apply(json.loads)

        tokens, starts, ends = self.model_pred(df['question'].to_list(), df['context'].to_list(), batch_size=args.evaluation_batch_size)
        start_indexes = starts.argsort(dim=-1, descending=True)[:, :self.n_best]
        end_indexes = ends.argsort(dim=-1, descending=True)[:, :self.n_best]

        references = []
        predictions = []
        for i, (_id, context, answers) in df[['id', 'context', 'answers']].iterrows():
            min_null_score = starts[i][0] + ends[i][0] # 0 is CLS Token
            start_context = tokens['input_ids'][i].tolist().index(self.tokenizer.sep_token_id)

            offset = tokens['offset_mapping'][i]
            valid_answers = []
            for start_index in start_indexes[i]:
                # Don't consider answers that are in questions
                if start_index<start_context:
                    continue
                for end_index in end_indexes[i]:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (start_index >= len(offset) or end_index >= len(offset)
                        or offset[start_index] is None or offset[end_index] is None):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or (end_index-start_index+1) > args.max_answer_length:
                        continue

                    start_char = offset[start_index][0]
                    end_char = offset[end_index][1]
                    valid_answers.append({"score": (starts[i][start_index] + ends[i][end_index]).item(),
                                            "text": context[start_char: end_char]})

            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                best_answer = {"text": "", "score": min_null_score}

            predictions.append({
                'id': str(_id),
                'prediction_text': best_answer['text'] if best_answer["score"] >= min_null_score else "",
                'no_answer_probability': min_null_score.item()/2
            })

            references.append({
                'id': str(_id),
                'answers': answers
            })

        with open('predictions.json', 'w') as f:
            json.dump(predictions, f, indent=4)

        with open('references.json', 'w') as f:
            json.dump(references, f, indent=4)

        metric = evaluate.load("squad_v2")
        return metric.compute(predictions=predictions, references=references)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_checkpoint", required=True, type=str, help="The name of model to evaluate.")
    parser.add_argument("--test_file", required=True, type=str, help="The input test file in csv.")

    parser.add_argument("--device", default='cuda:0', type=str, help="Device used for training.")
    parser.add_argument("--n_best", default=10, type=int, help="The total number of n-best predictions to generate.")
    parser.add_argument("--max_length", default=512, type=int, help="The maximum length of combination of question and context.")
    parser.add_argument('--doc_stride', default=256, type=int, help="The authorized overlap between two part of the context when splitting it is needed.")
    parser.add_argument('--max_answer_length', default=128, type=int, help="The authorized overlap between two part of the context when splitting it is needed.")
    parser.add_argument("--evaluation_batch_size", default=1, type=int, help="Batch size for evaluation.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_checkpoint)

    predictor = AnswerEvaluator(args, model, tokenizer)
    results = predictor.evaluate(args)
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
