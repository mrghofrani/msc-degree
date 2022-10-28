import random
import logging

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import jiwer
from tomark import Tomark
import pytorch_lightning as pl

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO) #TODO: change print with logger
logger = logging.getLogger()


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout):
        super().__init__()

        self.embedding = nn.Embedding(input_dim, emb_dim)

        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional = True)

        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):

        #src = [src len, batch size]
        #src_len = [batch size]

        embedded = self.dropout(self.embedding(src))

        #embedded = [src len, batch size, emb dim]

        #need to explicitly put lengths on cpu!
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_len.to('cpu'))

        packed_outputs, hidden = self.rnn(packed_embedded)

        #packed_outputs is a packed sequence containing all hidden states
        #hidden is now from the final non-padded element in the batch

        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        #outputs is now a non-packed sequence, all hidden states obtained
        #  when the input is a pad token are all zeros

        #outputs = [src len, batch size, hid dim * num directions]
        #hidden = [n layers * num directions, batch size, hid dim]

        #hidden is stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer

        #hidden [-2, :, : ] is the last of the forwards RNN
        #hidden [-1, :, : ] is the last of the backwards RNN

        #initial decoder hidden is final hidden state of the forwards and backwards
        #  encoder RNNs fed through a linear layer
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)))

        #outputs = [src len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()

        self.attn = nn.Linear((enc_hid_dim * 2) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias = False)

    def forward(self, hidden, encoder_outputs, mask):

        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]

        batch_size = encoder_outputs.shape[1]
        src_len = encoder_outputs.shape[0]

        #repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #hidden = [batch size, src len, dec hid dim]
        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim = 2)))

        #energy = [batch size, src len, dec hid dim]

        attention = self.v(energy).squeeze(2)

        #attention = [batch size, src len]

        attention = attention.masked_fill(mask == 0, -1e10)

        return F.softmax(attention, dim = 1)


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)

        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)

        self.fc_out = nn.Linear((enc_hid_dim * 2) + dec_hid_dim + emb_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, encoder_outputs, mask):

        #input = [batch size]
        #hidden = [batch size, dec hid dim]
        #encoder_outputs = [src len, batch size, enc hid dim * 2]
        #mask = [batch size, src len]

        input = input.unsqueeze(0)

        #input = [1, batch size]

        embedded = self.dropout(self.embedding(input))

        #embedded = [1, batch size, emb dim]

        a = self.attention(hidden, encoder_outputs, mask)

        #a = [batch size, src len]

        a = a.unsqueeze(1)

        #a = [batch size, 1, src len]

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        #encoder_outputs = [batch size, src len, enc hid dim * 2]

        weighted = torch.bmm(a, encoder_outputs)

        #weighted = [batch size, 1, enc hid dim * 2]

        weighted = weighted.permute(1, 0, 2)

        #weighted = [1, batch size, enc hid dim * 2]

        rnn_input = torch.cat((embedded, weighted), dim = 2)

        #rnn_input = [1, batch size, (enc hid dim * 2) + emb dim]

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        #output = [seq len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]

        #seq len, n layers and n directions will always be 1 in this decoder, therefore:
        #output = [1, batch size, dec hid dim]
        #hidden = [1, batch size, dec hid dim]
        #this also means that output == hidden
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim = 1))

        #prediction = [batch size, output dim]

        return prediction, hidden.squeeze(0), a.squeeze(1)


class AttnEncoderDecoder(pl.LightningModule):
    def __init__(self, encoder, decoder,
                 grapheme_characterizer, phoneme_characterizer,
                 teacher_forcing_ratio, max_length):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_characterizer = grapheme_characterizer
        self.trg_characterizer = phoneme_characterizer
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.src_pad_idx = grapheme_characterizer.pad_token_id
        self.trg_sos_idx = self.trg_characterizer.sos_token_id
        self.max_length = max_length

    def create_mask(self, src):
        mask = (src != self.src_pad_idx).permute(1, 0)
        return mask

    def forward(self, src, src_len, trg, teacher_forcing_ratio = 0.5):

        #src = [src len, batch size]
        #src_len = [batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use teacher forcing 75% of the time

        batch_size = src.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        #encoder_outputs is all hidden states of the input sequence, back and forwards
        #hidden is the final forward and backward hidden states, passed through a linear layer
        encoder_outputs, hidden = self.encoder(src, src_len)

        #first input to the decoder is the  tokens
        input = trg[0,:]

        mask = self.create_mask(src)

        #mask = [batch size, src len]
        outputs[0] = torch.as_tensor([self.trg_characterizer.sos_token_id]*self.trg_characterizer.n_chars)

        for t in range(1, trg_len):

            #insert input token embedding, previous hidden state, all encoder hidden states
            #  and mask
            #receive output tensor (predictions) and new hidden state
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs

    def training_step(self, batch, batch_idx):
        grapheme, phoneme = batch
        grapheme = grapheme.permute(1, 0)
        phoneme = phoneme.permute(1, 0)
        src_len = torch.as_tensor([len(x) for x in grapheme.permute(1,0)]).to(self.device)
        output_phoneme = self.forward(src=grapheme,
                                      src_len=src_len,
                                      trg=phoneme)

        output_dim = output_phoneme.shape[-1]

        phoneme = phoneme[1:].reshape(-1)
        output_phoneme = output_phoneme[1:].view(-1, output_dim)
        loss = F.cross_entropy(output_phoneme, phoneme, ignore_index=0)

        self.logger.experiment.add_scalars("Loss", {"train": loss}, global_step=self.current_epoch)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def generate(self, src):
        # src = [batch_size, src len]
        batch_size = src.shape[0]
        src_len = torch.as_tensor([len(x) for x in src]).to(self.device)

        src = src.permute(1,0)
        with torch.no_grad():
            encoder_outputs, hidden = self.encoder(src, src_len)

        mask = self.create_mask(src)

        # pred = [batch_size, pred length]
        pred = torch.zeros(batch_size).unsqueeze(dim=1).to(self.device).to(torch.int64)
        pred += self.trg_sos_idx

        for _ in range(self.max_length):
            # output = [batch_size, pred length, vocab size]
            # pred_token_ids = [batch_size, 1]
            with torch.no_grad():
                output, hidden, attention = self.decoder(pred[:,-1], hidden, encoder_outputs, mask)
            pred_token_ids = output.argmax(dim=1).unsqueeze(dim=1)
            pred = torch.cat((pred, pred_token_ids), dim=1)

        return pred

    def convert_token_ids_to_string(self, trg, pred):
        n_samples = len(trg)

        str_trg, str_pred = list(), list()
        for i in range(n_samples):
            _pred = pred[i].to('cpu').numpy()
            _trg = trg[i].to('cpu').numpy()

            str_pred.append(self.trg_characterizer.decode(_pred, ignore_special_tokens=True))
            str_trg.append(self.trg_characterizer.decode(_trg, ignore_special_tokens=True))

        return str_trg, str_pred

    def calculate_accuracy(self, trg, pred):
        n_samples = len(trg)
        true_ins = 0
        for i in range(n_samples):
            if trg[i] == pred[i]:
                true_ins += 1
        return true_ins / n_samples

    def accumulate_outputs(self, outputs):
        trg, pred = outputs[0]
        n_outputs = len(outputs)

        for i in range(1, n_outputs):
            _trg, _pred = outputs[i]
            trg = torch.cat((trg, _trg), dim=0)
            pred = torch.cat((pred, _pred), dim=0)

        return trg, pred

    def calculate_per_error(self, trg, pred):
        return jiwer.cer(trg, pred)

    def _shared_test_val_epoch_end(self, outputs, name):
        trg, pred = self.accumulate_outputs(outputs)
        trg, pred = self.convert_token_ids_to_string(trg, pred)
        acc = self.calculate_accuracy(trg, pred)
        per = jiwer.cer(trg, pred)

        sample = [{"Target": t, "Pred": p} for t, p in zip(trg[:10], pred[:10])]
        self.logger.experiment.add_text(f'Generated Samples Step {self.current_epoch} for {name}',
                                            Tomark.table(sample))
        self.logger.experiment.add_scalars("Acc", {name: acc}, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("PER", {name: per}, global_step=self.current_epoch)
        return acc, per

    def _shared_test_val_step(self, batch, batch_idx):
        src, trg = batch
        pred = self.generate(src)
        return trg, pred

    def validation_epoch_end(self, outputs):
        return self._shared_test_val_epoch_end(outputs, name="val")

    def test_epoch_end(self, outputs):
        return self._shared_test_val_epoch_end(outputs, name="test")

    def test_step(self, batch, batch_idx):
        return self._shared_test_val_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._shared_test_val_step(batch, batch_idx)
