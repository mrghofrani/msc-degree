import logging
from operator import index
from regex import P

import torch
from torch.nn import ConstantPad1d
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO) #TODO: change print with logger
logger = logging.getLogger()


class Characterizer:
    def __init__(self):
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.special_tokens = [self.pad_token_id,
                               self.sos_token_id,
                               self.eos_token_id]
        self.char2index = {"-": 0, "\\": 1, "/": 2, "?": 3}
        self.index2char = {0: "-", 1: "\\", 2: "/", 3: "?"}
        self.n_chars = 4

    def add_word(self, word):
        for char in word:
            self.add_char(char)

    def add_char(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.index2char[self.n_chars] = char
            self.n_chars += 1

    def decode(self, ids, ignore_special_tokens=False):
        return ''.join(['' if (ignore_special_tokens and (id in self.special_tokens))
                        else self.index2char[id] for id in ids])


class Grapheme2PhonemeDataset(Dataset):
    def __init__(
        self,
        name,
        tsv_file,
        grapheme_characterizer,
        phoneme_characterizer,
        max_length,
        pad_token_id,
        sos_token_id,
        eos_token_id,
        unk_token_id
    ):
        self.name = name
        self.max_length = max_length
        self.sos_token_id = sos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id
        self.pad_token_id = pad_token_id
        self.grapheme_characterizer = grapheme_characterizer
        self.phoneme_characterizer = phoneme_characterizer

        logger.info(f"Reading {self.name} dataset...")
        pairs = self.read_dataset(tsv_file)
        logger.info(f"Read {len(pairs)} word pairs")
        pairs = self.filter_pairs(pairs)
        logger.info(f"Trimmed to {len(pairs)} word pairs")
        logger.info("Counting words...")
        if name != "test":
            for pair in pairs:
                grapheme_characterizer.add_word(pair[0])
                phoneme_characterizer.add_word(pair[1])
        logger.info("Counted characters:")
        logger.info(f'Unique number of input characters: {grapheme_characterizer.n_chars}')
        logger.info(f'Unique number of input characters: {phoneme_characterizer.n_chars}')

        self.n_grapheme_characters = grapheme_characterizer.n_chars
        self.n_phoneme_characterizers = phoneme_characterizer.n_chars
        logger.info("Converting to tensors...")
        self.pairs = [self.tensorsFromPair(pair) for pair in pairs]

    def _filter_pair(self, p):
        return len(p[0].split(' ')) < self.max_length and \
            len(p[1].split(' ')) < self.max_length

    def filter_pairs(self, pairs):
        return [pair for pair in pairs if self._filter_pair(pair)]

    def read_dataset(self, tsv_file):
        with open(tsv_file) as f:
            lines = f.read().strip().split('\n')

        # Split every line into pairs and normalize
        pairs = [[s for s in l.split('\t')] for l in lines]
        pairs = [list(p) for p in pairs]

        return pairs

    def _indexes_from_word(self, charaterizer, word):
        return [charaterizer.char2index.get(char, self.unk_token_id) for char in word]

    def _tensor_from_word(self, characterizer, word):
        indexes = self._indexes_from_word(characterizer, word)
        indexes.insert(0, self.sos_token_id)
        indexes.append(self.eos_token_id)
        return torch.tensor(indexes, dtype=torch.int64)# TODO: .view(-1, 1) #TODO: convert to int

    def tensorsFromPair(self, pair):
        input_tensor = self._tensor_from_word(self.grapheme_characterizer, pair[0])
        target_tensor = self._tensor_from_word(self.phoneme_characterizer, pair[1])
        return (input_tensor, target_tensor)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx][0], self.pairs[idx][1]


def collate_fn(data, max_length):
    def pad_sequences(sequences, padding_value):
        sequences[0] = torch.cat((sequences[0], torch.zeros(max_length - sequences[0].shape[0])), dim=0)
        o = pad_sequence(sequences, padding_value=padding_value).transpose(0, 1).to(torch.int64)
        return o

    grapheme_seq, phoneme_seq = zip(*data)
    return (pad_sequences(list(grapheme_seq), padding_value=0),
            pad_sequences(list(phoneme_seq), padding_value=0))
