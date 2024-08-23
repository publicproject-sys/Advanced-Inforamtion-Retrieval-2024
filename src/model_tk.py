from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import GELU

import numpy as np
import pandas as pd
import math
import os

from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.data.dataloader import PyTorchDataLoader


class MultiHeadAttentionBlock(nn.Module):

    def __init__(
        self, embed_size, heads, word_embedding_dim, max_len, batch_size, debug=False
    ):

        super(MultiHeadAttentionBlock, self).__init__()

        self.embed_size = embed_size
        self.word_embedding_dim = word_embedding_dim
        self.heads = heads
        self.max_len = max_len
        assert (
            self.embed_size % self.heads == 0
        ), "Embed size needs to be divisible by number heads"
        self.head_dim = self.embed_size // self.heads
        self.batch_size = batch_size
        self.debug = debug

        self.Q = nn.Linear(self.embed_size, self.embed_size)
        self.K = nn.Linear(self.embed_size, self.embed_size)
        self.V = nn.Linear(self.embed_size, self.embed_size)
        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def attention(self, q, k, v, mask):
        """
        Computes the attention scores.

        Input: q, k, v (batch_size, heads, max_len, head_dim)
        Output: attention scores (batch_size, heads, max_len, head_dim)
        """

        # attention_scores = (Q * K^T) / sqrt(d_k)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # add heads and head_dim dimensions
            mask = mask.expand_as(attn_scores)
            masked_score = attn_scores.masked_fill(mask == 0, -1e10)
            if self.debug:
                print(f"MultiheadAttentionBlock mask shape: {mask.shape}")
        else:
            masked_score = attn_scores
        # dim attn_scores: (batch_size, heads, max_len, max_len)
        attn_scores_softmax = torch.softmax(
            masked_score, dim=-1
        )  # dim = -1 means last dimension meaning max_len
        # dim attn_scores_softmax: (batch_size, heads, max_len, max_len)

        if self.debug:
            print(
                f"MultiheadAttentionBlock attn_scores_softmax shape: {attn_scores_softmax.shape}"
            )
            print(f"MultiheadAttentionBlock v shape: {v.shape}")
        out = torch.matmul(
            attn_scores_softmax, v
        )  # swap max_len and head_dim #.transpose(-2, -1)
        # dim out: (batch_size, heads, max_len, head_dim)
        return out

    def separate_heads(self, input):
        """
        Reshapes the input tensor to separate the heads.

        Input: (batch_size, max_len, embed_size)
        Output: (batch_size, max_len, heads, head_dim)
        """

        out = input.reshape(self.batch_size, self.heads, self.max_len, -1)
        if self.debug:
            print(f"MultiheadAttentionBlock separated_heads shape: {out.shape}")
        # return shape: (batch_size, heads, max_len, head_dim)
        return out

    def forward(self, input, mask):

        # project word embeddings to embedding size (internal dim)
        q = self.Q(input)
        k = self.K(input)
        v = self.V(input)

        # apply head separation
        q = self.separate_heads(q)
        k = self.separate_heads(k)
        v = self.separate_heads(v)

        # attention
        attantione_PICKPOCKETS = self.attention(q, k, v, mask)
        # dim attantione_PICKPOCKETS: (batch_size, heads, max_len, head_dim)

        # concatenate heads
        attantione_PICKPOCKETS = attantione_PICKPOCKETS.permute(0, 2, 1, 3).reshape(
            self.batch_size, -1, self.embed_size
        )
        # dim attantione_PICKPOCKETS: (batch_size, max_len, embed_size)

        output = self.fc_out(attantione_PICKPOCKETS)
        return output


class FCBlock(nn.Module):

    def __init__(self, embed_size, dropout):
        super(FCBlock, self).__init__()
        self.embed_size = embed_size
        self.dropout = dropout
        self.fc = nn.Linear(self.embed_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout)
        self.gelu = GELU()

    def forward(self, input):
        out = self.fc(input)
        out = self.dropout(out)
        out = self.gelu(out)
        return out


class AddNormBlock(nn.Module):

    def __init__(self, embed_size, dropout):
        super(AddNormBlock, self).__init__()
        self.embed_size = embed_size
        self.dropout = dropout
        self.norm = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, input, skip):
        out = self.norm(input + skip)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_size,
        heads,
        dropout,
        word_embedding_dim,
        max_len,
        batch_size,
        debug=False,
    ):
        super(TransformerBlock, self).__init__()
        self.batch_size = batch_size
        self.debug = debug
        self.embed_size = embed_size
        self.heads = heads
        self.dropout = dropout
        self.word_embedding_dim = word_embedding_dim
        self.max_len = max_len
        self.multi_head_attention_block = MultiHeadAttentionBlock(
            self.embed_size,
            self.heads,
            self.word_embedding_dim,
            self.max_len,
            self.batch_size,
            self.debug,
        )
        self.add_norm_block1 = AddNormBlock(self.embed_size, self.dropout)
        self.fc_block = FCBlock(self.embed_size, self.dropout)
        self.add_norm_block2 = AddNormBlock(self.embed_size, self.dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, input, mask):
        if self.debug:
            print(f"TransformerBlock IN shape: {input.shape}")
        att = self.multi_head_attention_block(input, mask)
        skip1 = self.add_norm_block1(input, att)
        fcc = self.fc_block(skip1)
        skip2 = self.add_norm_block2(skip1, fcc)
        out = input * self.alpha + skip2 * (1 - self.alpha)
        if self.debug:
            print(f"TransformerBlock OUT shape: {out.shape}")
        return out


class PositionalEncodingBlock(nn.Module):

    def __init__(self, emb_size, dropout, max_len, debug=False):
        super().__init__()

        self.debug = debug
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size)
        )
        pe = torch.zeros(max_len, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if self.debug:
            print(f"PE IN shape: {x.shape}")
        x = x.permute(1, 0, 2)
        x = x + self.pe[: x.size(0)]
        x = self.dropout(x)
        if self.debug:
            print(f"PE OUT shape: {x.shape}")
        return x.permute(0, 1, 2)


class TransformerTK(nn.Module):

    def __init__(
        self,
        embed_size,
        heads,
        num_layers,
        word_embedding_dim,
        max_len,
        dropout,
        batch_size,
        debug=False,
    ):
        super(TransformerTK, self).__init__()
        self.debug = debug
        self.embed_size = embed_size
        self.heads = heads
        self.num_layers = num_layers
        self.word_embedding_dim = word_embedding_dim
        self.max_len = max_len
        self.dropout = dropout
        self.batch_size = batch_size
        self.linear = nn.Linear(self.word_embedding_dim, self.embed_size)
        self.positional_encoding_block = PositionalEncodingBlock(
            self.embed_size, self.dropout, self.max_len
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    self.embed_size,
                    self.heads,
                    self.dropout,
                    self.word_embedding_dim,
                    self.max_len,
                    self.batch_size,
                    self.debug,
                )
                for _ in range(self.num_layers)
            ]
        )

    def forward(self, input, mask):
        out = self.linear(input)
        for i, transformer_block in enumerate(self.transformer_blocks):
            if i == -1:
                out = self.positional_encoding_block(input)
            out = transformer_block(out, mask)
        # dim out: (batch_size, max_len, embed_size)
        return out


class TK(nn.Module):

    def __init__(
        self,
        n_kernels,
        max_len,
        n_heads,
        num_layers,
        hidden_dim,
        word_embedding_dim,
        dropout,
        batch_size,
        debug=False,
    ):

        super(TK, self).__init__()
        self.debug = debug
        self.n_kernels = n_kernels
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.word_embedding_dim = word_embedding_dim
        self.dropout = dropout
        self.batch_size = batch_size

        self.linear_length_path = nn.Linear(self.n_kernels, 1, bias=False)
        self.linear_log_path = nn.Linear(self.n_kernels, 1, bias=False)
        self.combine_paths = nn.Linear(2, 1, bias=False)

        # init linear scoring
        torch.nn.init.uniform_(
            self.linear_length_path.weight, -0.01, 0.01
        )  # inits taken from matchzoo
        torch.nn.init.uniform_(self.linear_log_path.weight, -0.01, 0.01)
        torch.nn.init.uniform_(self.combine_paths.weight, -0.01, 0.01)

        mu = torch.FloatTensor(self.kernel_mus(self.n_kernels)).view(
            1, 1, 1, self.n_kernels
        )
        sigma = torch.FloatTensor(self.kernel_sigmas(self.n_kernels)).view(
            1, 1, 1, self.n_kernels
        )

        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

        self.tk_transformer = TransformerTK(
            embed_size=self.hidden_dim,
            heads=n_heads,
            num_layers=num_layers,
            word_embedding_dim=self.word_embedding_dim,
            max_len=self.max_len,
            dropout=self.dropout,
            batch_size=self.batch_size,
            debug=self.debug,
        )

    def forward(self, query_batch, doc_batch, query_mask, doc_mask):

        query_batch = self.tk_transformer(query_batch, query_mask)
        doc_batch = self.tk_transformer(doc_batch, doc_mask)
        ## cosine similarity matrix

        # create mask for padded elements to exclude them from the kernel scores
        mask = torch.bmm(query_mask.unsqueeze(2), doc_mask.unsqueeze(1))
        mask = mask.unsqueeze(3)

        # ------------------ Match matrix ------------------
        # calculate cosine similarity scores between each pair of query and document words
        norm_q = torch.sqrt(torch.sum(torch.square(query_batch), 2, keepdim=True))
        normalized_q_embed = query_batch / norm_q
        norm_d = torch.sqrt(torch.sum(torch.square(doc_batch), 2, keepdim=True))
        normalized_d_embed = doc_batch / norm_d
        # normalized_d_embed = normalized_d_embed.transpose(1, 2)
        match_matrix = torch.bmm(normalized_q_embed, normalized_d_embed.transpose(1, 2))
        # fill nan elements with 0
        match_matrix[match_matrix != match_matrix] = 0
        # match_matrix shape: (batch, max_len, max_len, 1)
        match_matrix = match_matrix.unsqueeze(3)

        kernel_scores = torch.exp(
            -torch.square(torch.sub(match_matrix, self.mu))
            / (torch.mul(torch.square(self.sigma), 2))
        )

        # apply mask by multiplying kernel scores with the mask
        kernel_scores_masked = kernel_scores * mask
        # sum along the document dimension
        # input shape: (batch, query_max, doc_max, n_kernels); output shape: (batch, query_max, n_kernels)

        # kernels per query
        kernel_scores_query = torch.sum(kernel_scores_masked, dim=2)

        # log normalization path
        log_kernel_scores_query = (
            torch.log(torch.clamp(kernel_scores_query, min=1e-10)) * 0.01
        )
        log_kernel_scores_query_masked = log_kernel_scores_query * query_mask.unsqueeze(
            -1
        )
        log_path_kernel_scores = log_kernel_scores_query_masked.sum(dim=1)

        # length normalization path
        length = (
            torch.sum(doc_mask, dim=1, keepdim=True)
            .unsqueeze(-1)
            .expand_as(kernel_scores_query)
            * 0.01
        )
        kernel_scores_length_normalized = kernel_scores_query / length
        kernel_scores_length_normalized_masked = (
            kernel_scores_length_normalized * query_mask.unsqueeze(-1)
        )
        length_norm_path_kernel_scores = kernel_scores_length_normalized_masked.sum(
            dim=1
        )
        length_norm_path_kernel_scores_linear = self.linear_length_path(
            length_norm_path_kernel_scores
        )
        log_path_kernel_scores_linear = self.linear_log_path(log_path_kernel_scores)

        # combine paths
        combined_scores = torch.cat(
            (length_norm_path_kernel_scores_linear, log_path_kernel_scores_linear),
            dim=1,
        )
        scores = self.combine_paths(combined_scores)

        return scores

    def kernel_mus(self, n_kernels: int):
        """
        get the mu for each guassian kernel. Mu is the middle of each bin
        :param n_kernels: number of kernels (including exact match). first one is exact match
        :return: l_mu, a list of mu.
        """
        l_mu = [1.0]
        if n_kernels == 1:
            return l_mu

        bin_size = 2.0 / (n_kernels - 1)  # score range from [-1, 1]
        l_mu.append(1 - bin_size / 2)  # mu: middle of the bin
        for i in range(1, n_kernels - 1):
            l_mu.append(l_mu[i] - bin_size)
        return l_mu

    def kernel_sigmas(self, n_kernels: int):
        """
        get sigmas for each guassian kernel.
        :param n_kernels: number of kernels (including exactmath.)
        :param lamb:
        :param use_exact:
        :return: l_sigma, a list of simga
        """
        bin_size = 2.0 / (n_kernels - 1)
        l_sigma = [0.0001]  # for exact match. small variance -> exact match
        if n_kernels == 1:
            return l_sigma

        l_sigma += [0.5 * bin_size] * (n_kernels - 1)
        return l_sigma
