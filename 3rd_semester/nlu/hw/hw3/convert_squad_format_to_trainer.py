import json
import argparse
import pandas as pd


def read_dataset(path):
    with open(path) as f:
        data = json.load(f)

    dataset = []
    for document in data['data']:
        title = document['title']
        for paragraph in document['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                _qa = dict()
                _qa['id'] = qa['id']
                _qa['question'] = qa['question']
                _qa['answers'] = {
                    'answer_start': [ans['answer_start'] for ans in qa['answers']],
                    'text': [ans['text'] for ans in qa['answers']]}
                _qa['impossible'] = qa['is_impossible']
                _qa['context'] = context
                _qa['title'] = title

                dataset.append(_qa)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_model", required=True, type=str, help="The name of pretrained model from huggingface.")
    parser.add_argument("--train_file", required=True, type=str, help="The input training file in squad format.")
    parser.add_argument("--validation_file", required=True, type=str, help="The input evaluation file in squad format.")
    parser.add_argument("--test_file", required=True, type=str, help="The input test file in squad format.")

    parser.add_argument("--train_output_file", required=True, type=str, help="The input training file in squad format.")
    parser.add_argument("--validation_output_file", required=True, type=str, help="The input evaluation file in squad format.")
    parser.add_argument("--test_output_file", required=True, type=str, help="The input test file in squad format.")
    args = parser.parse_args()

    train = read_dataset(args.train_file)
    validation = read_dataset(args.validation_file)
    test = read_dataset(args.test_file)

    train_df = pd.DataFrame(train)
    train_df['answers'] = train_df['answers'].apply(json.dumps)
    train_df.to_csv(args.train_output_file, index=False)

    validation_df = pd.DataFrame(validation)
    validation_df['answers'] = validation_df['answers'].apply(json.dumps)
    validation_df.to_csv(args.validation_output_file, index=False)

    test_df = pd.DataFrame(test)
    test_df['answers'] = test_df['answers'].apply(json.dumps)
    test_df.to_csv(args.test_output_file, index=False)
