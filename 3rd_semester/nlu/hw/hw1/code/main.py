import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import logging
from functools import partial

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from data import Characterizer, Grapheme2PhonemeDataset, collate_fn
from encoder_decoder import Encoder, Decoder, AttnEncoderDecoder, Attention
from transformer import Transformer, TransEncoder, TransDecoder


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO) #TODO: change print with logger
logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description='''
                                     Code for Homework 1 of NLU course. This code is a wrapper for both
                                     Encoder-Decoder with Attention and Transformer models.
                                     ''')
    parser.add_argument('--train_dataset', type=str, required=True,
                        help='Data should be in TSV format.')
    parser.add_argument('--test_dataset', type=str, required=True,
                        help='Data should be in TSV format.')
    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--val_batch_size', type=int, default=256)
    parser.add_argument('--test_batch_size', type=int, default=256)
    parser.add_argument('--train_val_split', type=float, default=0.8)
    parser.add_argument('--model', type=str, choices=['encoder_decoder', 'transformer'], default='encoder_decoder')

    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--dropout_p', type=float, default=0.1)
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--max_emb_length', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=5)
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5)

    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--ffn_hidden_size', type=int, default=512)

    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    logger.info("Reading datasets...")
    grapheme_characterizer = Characterizer()
    phoneme_characterizer = Characterizer()
    train_dataset = Grapheme2PhonemeDataset(name="train",
                                            tsv_file=args.train_dataset,
                                            grapheme_characterizer=grapheme_characterizer,
                                            phoneme_characterizer=phoneme_characterizer,
                                            max_length=args.max_length,
                                            pad_token_id = grapheme_characterizer.pad_token_id,
                                            sos_token_id = grapheme_characterizer.sos_token_id,
                                            eos_token_id = grapheme_characterizer.eos_token_id,
                                            unk_token_id = grapheme_characterizer.unk_token_id)
    test_dataset = Grapheme2PhonemeDataset(name="test",
                                           tsv_file=args.test_dataset,
                                           grapheme_characterizer=grapheme_characterizer,
                                           phoneme_characterizer=phoneme_characterizer,
                                           max_length=args.max_length,
                                           pad_token_id = grapheme_characterizer.pad_token_id,
                                           sos_token_id = grapheme_characterizer.sos_token_id,
                                           eos_token_id = grapheme_characterizer.eos_token_id,
                                           unk_token_id = grapheme_characterizer.unk_token_id)
    train_size = int(args.train_val_split * len(train_dataset))
    dev_size = len(train_dataset) - train_size
    train_dataset, dev_dataset = torch.utils.data.random_split(train_dataset, [train_size, dev_size],
                                                               generator=torch.Generator().manual_seed(42))

    logger.info("Creating dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=6,
                                  collate_fn=partial(collate_fn, max_length=args.max_length), shuffle=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=64, num_workers=6,
                                collate_fn=partial(collate_fn, max_length=args.max_length), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=6,
                                 collate_fn=partial(collate_fn, max_length=args.max_length), shuffle=False)

    if args.model == 'encoder_decoder':
        encoder = Encoder(input_dim=grapheme_characterizer.n_chars,
                          emb_dim=args.hidden_size,
                          enc_hid_dim=args.hidden_size,
                          dec_hid_dim=args.hidden_size,
                          dropout=args.dropout_p)
        attn = Attention(enc_hid_dim=args.hidden_size,
                         dec_hid_dim=args.hidden_size)
        decoder = Decoder(output_dim=phoneme_characterizer.n_chars,
                          emb_dim=args.hidden_size,
                          enc_hid_dim=args.hidden_size,
                          dec_hid_dim=args.hidden_size,
                          dropout=args.dropout_p,
                          attention=attn)
        model = AttnEncoderDecoder(encoder=encoder,
                                   decoder=decoder,
                                   grapheme_characterizer=grapheme_characterizer,
                                   phoneme_characterizer=phoneme_characterizer,
                                   teacher_forcing_ratio=args.teacher_forcing_ratio,
                                   max_length=args.max_length)
    else:
        enc = TransEncoder(input_dim=grapheme_characterizer.n_chars,
                      hid_dim=args.hidden_size,
                      n_layers=args.n_layers,
                      n_heads=args.n_heads,
                      pf_dim=args.ffn_hidden_size,
                      dropout=args.dropout_p,
                      device=args.device,
                      max_length=args.max_length)

        dec = TransDecoder(phoneme_characterizer.n_chars, args.hidden_size,
                      args.n_layers, args.n_heads, args.ffn_hidden_size,
                      args.dropout_p, args.device)

        model = Transformer(enc, dec,
                            grapheme_characterizer,
                            phoneme_characterizer,
                            args.max_length)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./logs")
    trainer = pl.Trainer(logger=tb_logger, accelerator=args.device,
                         max_epochs=args.n_epochs)
    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=dev_dataloader)
    trainer.test(model=model,
                 dataloaders=test_dataloader)

if __name__ == "__main__":
    main()
