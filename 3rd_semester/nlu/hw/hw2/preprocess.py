import os
from conllu import parse_incr


def load_and_precess(filename):
    dataset = list()
    with open(filename, encoding="utf-8") as f:
        for tl in parse_incr(f):
            dataset.append(tl)

    seq_in = list()
    seq_out = list()
    labels = list()
    slot_labels = set()
    intents = set()
    for sentence in dataset:
        labels.append(sentence[0]['lemma'])
        seq_in.append(" ".join([token['form'] for token in sentence]))
        seq_out.append(" ".join([token['upos'] if token['upos'] != "NoLabel" else "O" for token in sentence]))
        slot_labels = slot_labels.union(set([token['upos'] if token['upos'] != "NoLabel" else "O" for token in sentence]))
        intents.add(sentence[0]['lemma'])

    return seq_in, seq_out, labels, slot_labels, intents


def main():
    for filename in ['train', 'dev', 'test']:
        seq_in, seq_out, labels, slot_labels, intents = load_and_precess(f'data/{filename}-en.conllu')
        os.makedirs(f'data/{filename}', exist_ok=True)
        with open(f'data/{filename}/seq_in', 'w') as fi, open(f'data/{filename}/seq_out', 'w') as fo, open(f'data/{filename}/label', 'w') as fl:
            for i, o, l in zip(seq_in, seq_out, labels):
                fi.write(f"{i}\n")
                fo.write(f"{o}\n")
                fl.write(f"{l}\n")
        if filename == 'train':
            with open('data/slot_labels.txt', 'w') as f:
                for label in slot_labels:
                    f.write(f"{label}\n")
            with open('data/intents.txt', 'w') as f:
                for intent in intents:
                    f.write(f"{intent}\n")


if __name__ == "__main__":
    main()
