import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
        super().__init__()       
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len

        self.sos_token = torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f'Input sequence is longer than {self.seq_len} tokens')

        # Add SOS and EOS tokens to src text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),
                # torch.tensor(self.pad_token.repeat(enc_num_padding_tokens), dtype=torch.int64)
                # self.pad_token.repeat(enc_num_padding_tokens)
            ],
            dim=0
        )

        # Add SOS to tgt text
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
                # torch.tensor(self.pad_token.repeat(dec_num_padding_tokens), dtype=torch.int64)
                # self.pad_token.repeat(dec_num_padding_tokens)
            ],
            dim=0
        )

        # Difference decoder_input and label is the sos token vs the eos token
        # Add EOS to the label (what we expect the model to predict from the decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),
                # torch.tensor(self.pad_token.repeat(dec_num_padding_tokens), dtype=torch.int64)
                # self.pad_token.repeat(dec_num_padding_tokens)
            ],
            dim=0
        )

        # Pad or truncate to seq_len
        # if len(enc_input_tokens) < self.seq_len:
        #     enc_input_tokens = torch.cat((enc_input_tokens, self.pad_token.repeat(self.seq_len - len(enc_input_tokens))))
        # else:
        #     enc_input_tokens = enc_input_tokens[:self.seq_len]
        # # Add SOS and EOS tokens
        # enc_input_tokens = torch.cat((self.sos_token, enc_input_tokens, self.eos_token))

        # Tests
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, # (Seq_len)
            "decoder_input": decoder_input, # (Seq_len)
            # Should the value be int or float for False/True? Response: float. Why?
            # Also, why is dimensions (1, 1, Seq_len)? Because of the batch dimension?
            # Batch is first dimension. What is the 2nd dimension?
            # Response: 1. Why? Response: 1 because it's a single sentence.
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, Seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, Seq_Len) & (1, 1, Seq_Len)
            "label": label, # (Seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }

def causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int64)
    return mask == 0