from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable

from allennlp.modules.text_field_embedders import TextFieldEmbedder


class KNRM(nn.Module):

    def __init__(
        self,
        n_kernels: int,
        query_max: int = 30,
        doc_max: int = 180,
        hidden_dim: int = 128,
        word_embeddings_dim: int = 300,
    ):

        super(KNRM, self).__init__()

        self.n_kernels = n_kernels
        self.query_max = query_max
        self.doc_max = doc_max

        mu = torch.FloatTensor(self.kernel_mus(self.n_kernels)).view(
            1, 1, 1, self.n_kernels
        )
        sigma = torch.FloatTensor(self.kernel_sigmas(self.n_kernels)).view(
            1, 1, 1, self.n_kernels
        )

        self.embeddings_dim = word_embeddings_dim

        self.hidden_dim = hidden_dim

        self.register_buffer("mu", mu)
        self.register_buffer("sigma", sigma)

        self.linear = nn.Linear(self.embeddings_dim, hidden_dim)
        self.linear_scoring = nn.Linear(self.n_kernels, 1)
        self.tanh_scoring = nn.Tanh()

    def forward(
        self,
        query_emb,
        doc_emb,
        query_pad_mask,
        doc_pad_mask,
    ) -> torch.Tensor:

        debug = False

        # ------------------ Linear transformation ------------------
        # input shape: (batch, query_max, emb_dim); output shape: (batch, query_max, hidden_dim)
        query_emb = self.linear(query_emb)
        # input shape: (batch, doc_max, emb_dim); output shape: (batch, doc_max, hidden_dim)
        doc_emb = self.linear(doc_emb)

        if debug:
            print(f"query_embeddings shape: {query_emb.shape}")
            print(f"document_embeddings shape: {doc_emb.shape}")
            print(f"queru embeddings: {query_emb[0]}")
            print(f"document embeddings: {doc_emb[0]}")

        # create mask for padded elements to exclude them from the kernel scores
        mask = torch.bmm(query_pad_mask.unsqueeze(2), doc_pad_mask.unsqueeze(1))
        mask = mask.unsqueeze(3)

        # ------------------ Match matrix ------------------
        # calculate cosine similarity scores between each pair of query and document words
        norm_q = torch.sqrt(torch.sum(torch.square(query_emb), 2, keepdim=True))
        normalized_q_embed = query_emb / norm_q
        norm_d = torch.sqrt(torch.sum(torch.square(doc_emb), 2, keepdim=True))
        normalized_d_embed = doc_emb / norm_d
        match_matrix = torch.bmm(normalized_q_embed, normalized_d_embed.transpose(1, 2))
        # fill nan elements with 0
        match_matrix[match_matrix != match_matrix] = 0

        # match_matrix shape: (batch, query_max, doc_max, 1)
        match_matrix = match_matrix.unsqueeze(3)

        if debug:
            print(f"match_matrix shape: {match_matrix.shape}")

        # ------------------ Kernel pooling ------------------
        # match matrix shape (batch, query_max, doc_max, 1)
        # mu and sigma shape (1, 1, 1, n_kernels)
        # kernel_scores shape: (batch, query_max, doc_max, n_kernels)
        kernel_scores = torch.exp(
            -torch.square(torch.sub(match_matrix, self.mu))
            / (torch.mul(torch.square(self.sigma), 2))
        )

        if debug:
            print(f"kernel_scores shape: {kernel_scores.shape}")
            print(kernel_scores[0])

        # apply mask by multiplying kernel scores with the mask
        kernel_scores = kernel_scores * mask
        # sum along the document dimension
        # input shape: (batch, query_max, doc_max, n_kernels); output shape: (batch, query_max, n_kernels)
        kernel_scores = torch.sum(kernel_scores, dim=2)

        if debug:
            print(
                f"kernel_scores after summing along document words shape: {kernel_scores.shape}"
            )
            print(kernel_scores[0])

        # apply log transformation and scale the scores
        kernel_scores = torch.log(torch.clamp(kernel_scores, min=1e-10)) * 0.01

        if debug:
            print("kernel scores shape: ", kernel_scores.shape)
            print("query_pad_mask shape: ", query_pad_mask.shape)

        # apply mask by multiplying kernel scores with the mask
        kernel_scores = kernel_scores * query_pad_mask.unsqueeze(-1)

        # sum along the query dimension
        # input shape: (batch, query_max, n_kernels); output shape: (batch, n_kernels)
        features = kernel_scores.sum(dim=1)
        if debug:
            print(f"features shape: {features.shape}")

        # ------------------ Scoring ------------------
        # obtain scores by applying a linear layer followed by a tanh activation
        # input shape: (batch, n_kernels); output shape: (batch, 1)
        scores = self.tanh_scoring(self.linear_scoring(features))

        if debug:
            print(f"scores shape: {scores.shape}")

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


def knrm_training_loop(
    model, loader, optimizer, loss_criterion, batch_embedder, device, epochs=1
):
    for epoch in range(epochs):
        model.train()
        for idx, batch in enumerate(loader):
            optimizer.zero_grad()
            (
                query_emb,
                doc_pos_emb,
                doc_neg_emb,
                query_pad_mask,
                document_pad_mask_pos,
                document_pad_mask_neg,
            ) = batch_embedder(batch)

            # Move embeddings to the specified device
            query_emb = query_emb.to(device)
            doc_pos_emb = doc_pos_emb.to(device)
            doc_neg_emb = doc_neg_emb.to(device)
            query_pad_mask = query_pad_mask.to(device)
            document_pad_mask_pos = document_pad_mask_pos.to(device)
            document_pad_mask_neg = document_pad_mask_neg.to(device)

            pred_pos = model(
                query_emb, doc_pos_emb, query_pad_mask, document_pad_mask_pos
            )
            pred_neg = model(
                query_emb, doc_neg_emb, query_pad_mask, document_pad_mask_neg
            )

            # Clear memory
            del (
                query_emb,
                doc_pos_emb,
                doc_neg_emb,
                query_pad_mask,
                document_pad_mask_pos,
                document_pad_mask_neg,
            )

            loss = loss_criterion(
                pred_pos, pred_neg, torch.ones(pred_pos.shape[0]).to(device)
            )
            loss.backward()
            optimizer.step()

            if idx % 50 == 0:
                print(f"{idx} Batch loss:", loss.item())

            # if idx * 256 >= 1000000:
            #     break

    return model
