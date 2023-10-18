import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeddings = nn.Embedding(vocab_size, d_model) # From the paper

    def forward(self, x) -> torch.Tensor:
        return self.embeddings(x) * math.sqrt(self.d_model)
    
    
class PositionalEncoding(nn.Module):
    # seq_len: maximum sequence length
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        ## What does register_buffer do here? 
        ## Answer: https://discuss.pytorch.org/t/what-is-the-difference-between-register-buffer-and-register-parameter-of-nn-module/32723
        ## register_buffer is used to register some persistent state in the model.
        ## The difference between register_buffer and register_parameter is that
        ## parameters are Tensors that are optimized during the training process
        ## while buffers are Tensors that are not optimized.
        ## This is useful when, for example, you have a running mean that you want
        ## to keep track of during the training.
        ## You want to keep it as a buffer instead of a parameter because you don't
        ## want to optimize it.
        ## Example: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        ## In the example, the positional encoding is not optimized during the training
        ## process, so it is registered as a buffer.
        ## The positional encoding is a tensor of shape (1, seq_len, d_model).
        ## The positional encoding is added to the input embeddings.
        ## The positional encoding is not updated during the training process.
        ## The positional encoding is not a parameter of the model.
        ## The positional encoding is a buffer of the model.
        ## The positional encoding is a persistent state of the model.
        ## All the above for this statement below, which we have commented out.
        # self.register_buffer("pe", self._get_positional_encodings())

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)


        ## Explain the formula below and the reason of optimization
        ## Answer: https://stackoverflow.com/questions/46452020/sinusoidal-embedding-attention-is-all-you-need
        ## The reason for the rearrangement of formula using log and exp is to avoid the
        ## computation of large numbers. The formula is same as the one in the paper. Here is the derivation:
        ## https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
        ## https://kikaben.com/transformers-positional-encoding/

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # Add a dimension at the beginning of the tensor for batch (1, seq_len, d_model)
        self.register_buffer('pe', pe) # Register the tensor as a buffer of the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.shape[1], :].requires_grad_(False)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__ (self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h

        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model) # Wk
        self.w_v = nn.Linear(d_model, d_model) # Wv

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout) -> torch.Tensor:
        d_k = query.shape[-1]

        # @ sign means matrix multiplication in PyTorch
        # Batch, h, seq_len, d_k --> Batch, h, seq_len, seq_len
        attention_scores = (query @ key.transpose(-2, -1))/math.sqrt(d_k)

        # Apply mask to the attention scores before softmax
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # Batch, h, seq_len, seq_len

        # Apply dropout to the attention scores
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores # Batch, h, seq_len, d_k

    def forward(self, q, k, v, mask = None):
        query = self.w_q(q) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2) # transpose because we want the shape to be (Batch, h, seq_len, d_k)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2) # transpose because we want the shape to be (Batch, h, seq_len, d_k)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2) # transpose because we want the shape to be (Batch, h, seq_len, d_k)

        # attention scores returned for visualization
        # x is Batch, h, seq_len, d_k
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        
        # Note the use of continuous() and view() here
        # Continuous() makes the tensor contiguous in memory
        # so that view() can be applied and the tensor can be reshaped
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) # Batch, seq_len, d_model

        return self.w_o(x) # Batch, seq_len, d_model

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    # What does sublayer represent?
    def forward(self, x, sublayer) -> torch.Tensor:
        # Unlike in the paper, sublayer is applied after normalization. In paper, normalization is applied after sublayer.
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        # Check now nn.ModuleList is used here
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        # self.residual_connection_1 = ResidualConnection(dropout)
        # self.residual_connection_2 = ResidualConnection(dropout)
    
    # src_mask applied to the input of the encoder
    # It helps with hiding the interactions of the padding word with other words
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        # Why one more normalization layer here?
        self.norm = LayerNormalization()

    def forward(self, x, src_mask):
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask) -> torch.Tensor:
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocal_list: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocal_list)

    def forward(self, x) -> torch.Tensor:
        # x is Batch, seq_len, d_model --> Batch, seq_len, vocal_list
        # why log_softmax, and not just softmax?
        # 
        return torch.log_softmax(self.proj(x), dim = -1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask) -> torch.Tensor:
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output, src_mask, tgt, tgt_mask) -> torch.Tensor:
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x) -> torch.Tensor:
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    # Ideally you dont need two separate positional encoding layers, as the positional encoding is the same for both source and target
    # But we are doing it here for the sake of clarity and teachability
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create the encoder and the decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer) # .to(dtype=torch.float16)
        #    .to(memory_efficient=True) \


    # Initialize the parameters of the transformer
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
