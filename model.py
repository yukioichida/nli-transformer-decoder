#!/usr/bin/env python3

import math
import torch
import torch.nn as nn


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class PositionWiseFF(nn.Module):
    def __init__(self, word_dim, hidden_dim, dropout):
        super(PositionWiseFF, self).__init__()
        self.input_dim = word_dim
        self.hidden_dim = hidden_dim
        self.ff_layer1 = nn.Linear(word_dim, hidden_dim)
        self.ff_layer2 = nn.Linear(hidden_dim, word_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.ff_layer1.weight, std=0.02)
        nn.init.normal_(self.ff_layer1.bias, 0)

        nn.init.normal_(self.ff_layer2.weight, std=0.02)
        nn.init.normal_(self.ff_layer2.bias, 0)

    def forward(self, x):
        hidden = gelu(self.ff_layer1(x))
        hidden = self.ff_layer2(hidden)
        return self.dropout(hidden)


class Attention(nn.Module):

    def __init__(self, word_dim, seq_len, n_heads, dropout, scale_attentions=True):
        super(Attention, self).__init__()
        self.seq_len = seq_len
        self.word_dim = word_dim
        self.n_heads = n_heads
        # Prevent to attend posterior words
        attention_mask = torch.tril((torch.ones(seq_len, seq_len))).view(1, 1, seq_len, seq_len)
        self.register_buffer("attention_mask", attention_mask)
        # Scale the attention if needed
        self.scale_attentions = scale_attentions
        # Convolves the input matrix in order to generate a greater matrix enough to split in 3 sub matrix : Q,K,V
        self.linear = nn.Linear(word_dim, word_dim * 3)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_output_layer = nn.Linear(word_dim, word_dim)
        self.output_dropout = nn.Dropout(dropout)

    def split_heads(self, x, permute=False):
        """
            Creates a new axis for attention heads
        :param x: tensor [batch_size, seq_len, word_dim]
        :param permute: whether should reorder the result axis
        :return: tensor [batch_size, seq_len, number of attention heads, word_dim]
        """
        # Including the head quantity in dimension, reducing the word_dim which is x.size(-1)
        new_x_shape = x.size()[:-1] + (self.n_heads, x.size(-1) // self.n_heads)
        x = x.view(*new_x_shape)
        # changing the shape of k to do matrix multiplication with q
        if permute:
            return x.permute(0, 2, 3, 1)  # for matrix multiplication with q in attention method (q*k)
        else:
            return x.permute(0, 2, 1, 3)

    def regroup_heads(self, x):
        """

        :param x: tensor [batch_size, seq_len, number of attention heads, word_dim]
        :return: tensor [batch_size, seq_len, word_dim]
        """
        x = x.permute(0, 2, 1, 3).contiguous()
        # first two axis + original word dim
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)

    def calc_attention(self, q, k, v):
        attn = torch.matmul(q, k)
        if self.scale_attentions:
            attn = attn / math.sqrt(v.size(-1))
        # adjust shape of the mask to fit into the input shape
        mask = self.attention_mask[:, :, :attn.size(-2), :attn.size(-1)]
        # remove attention to subsequent position
        attn = attn * mask
        # put -infinity into paddings in order to softmax ignore it
        negative_infinity = -1e9
        attn = attn + negative_infinity * (1 - attn)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        return torch.matmul(attn, v)

    def forward(self, x):
        # Triplicate the third axis (word dim)
        x = self.linear(x)
        # Split in the third axis, which is the word dimension
        query, key, value = x.split(self.word_dim, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, True)
        value = self.split_heads(value)

        attn = self.calc_attention(query, key, value)
        # back to the original shape
        attn = self.regroup_heads(attn)
        attn = self.attn_output_layer(attn)
        return self.output_dropout(attn)


class DecoderBlock(nn.Module):

    def __init__(self, seq_len, word_dim, n_heads, dropout=0.1, scale=False):
        super(DecoderBlock, self).__init__()
        self.attention = Attention(word_dim, seq_len, n_heads, dropout, scale_attentions=scale)
        self.layerNorm1 = nn.LayerNorm(word_dim)
        self.ff = PositionWiseFF(word_dim, word_dim * 4, dropout)
        self.layerNorm2 = nn.LayerNorm(word_dim)

    def forward(self, x):
        """

        :param x:
        :return: tensor [batch_size, seq_len, word_dim]
        """
        attn = self.attention(x)
        norm1 = self.layerNorm1(attn + x)  # residual
        ff_result = self.ff(norm1)
        hidden_state = self.layerNorm2(ff_result + norm1)  # residual
        return hidden_state


class TransformerDecoder(nn.Module):

    def __init__(self, vocab_size, max_seq_length, word_embedding_dim, n_layers, n_heads, output_dim,
                 eos_token, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        # Including positional encodings into word embedding matrix and eos_token
        qtty_word_embedding = vocab_size + max_seq_length + 1
        self.word_embedding_dim = word_embedding_dim
        self.embeddings = nn.Embedding(qtty_word_embedding, word_embedding_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.decoder_layers = nn.ModuleList([DecoderBlock(max_seq_length, word_embedding_dim, n_heads, dropout)
                                             for _ in range(n_layers)])
        self.output_layer = nn.Linear(word_embedding_dim, output_dim)
        self.eos_token = eos_token

        nn.init.normal_(self.output_layer.weight, std=0.02)
        nn.init.normal_(self.output_layer.bias, 0)

    def forward(self, x):
        """
            OBS: the last axis informs whether is the word sequence (pos_index = 0)
            or is the position index(pos_index = 1)
        :param x: tensor [batchsize, max_len, pos_index]
        :return: tensor [batchsize, num_classes]
        """
        # x = x.view(-1, x.size(-2), x.size(-1)) # SNLI only have one input (concatenated p and h)
        embeddings = self.embeddings(x)  # [batchsize, max_len, pos_index, word_dim]
        embeddings = self.input_dropout(embeddings)
        hidden = embeddings.sum(dim=2)  # sum the last axes = word vectors + positional vectors
        for decoder in self.decoder_layers:
            hidden = decoder(hidden)
        # Put all hidden states into a single vector
        hidden = hidden.view(-1, self.word_embedding_dim)
        # get the original indexes of sentence ...
        hidden_flat = x[..., 0].contiguous().view(-1)
        # ...and lookup the hidden state that is in the end of sentence token
        last_hidden = hidden[hidden_flat == self.eos_token, :]
        # With the hidden state of the last token, apply linear transformation
        logits = self.output_layer(last_hidden)
        return logits
