from typing import Dict, Iterator, List

import torch
import torch.nn as nn

class BatchWordEmbedder(nn.Module):
    def __init__(self, word_embedder, device, query_max=128, doc_max=128, train = True):
        super(BatchWordEmbedder, self).__init__()
        self.word_embedder = word_embedder
        self.query_max = query_max
        self.doc_max = doc_max
        self.device = device
        self.train_ = train

    def forward(self, batch):
        if self.train_:
            query = batch["query_tokens"]#["tokens"]["tokens"]
            doc_pos = batch["doc_pos_tokens"]#["tokens"]["tokens"]
            doc_neg = batch["doc_neg_tokens"]#["tokens"]["tokens"]
            
            query = torch.nn.functional.pad(query["tokens"]["tokens"], (0, self.query_max - query["tokens"]["tokens"].size(1)), value=0)
            doc_pos = torch.nn.functional.pad(doc_pos["tokens"]["tokens"], (0, self.doc_max - doc_pos["tokens"]["tokens"].size(1)), value=0)
            doc_neg = torch.nn.functional.pad(doc_neg["tokens"]["tokens"], (0, self.doc_max - doc_neg["tokens"]["tokens"].size(1)), value=0)
            
            query_pad_mask = (query > 1).float() # > 1 to also mask oov terms
            document_pad_mask_pos = (doc_pos > 1).float()
            document_pad_mask_neg = (doc_neg > 1).float()

            query = {"tokens": {"tokens": query}}
            doc_pos = {"tokens": {"tokens": doc_pos}}
            doc_neg = {"tokens": {"tokens": doc_neg}}

            query_emb = self.word_embedder(query).to(self.device)
            doc_pos_emb = self.word_embedder(doc_pos).to(self.device)
            doc_neg_emb = self.word_embedder(doc_neg).to(self.device)
            query_pad_mask = query_pad_mask.to(self.device)
            document_pad_mask_pos = document_pad_mask_pos.to(self.device)
            document_pad_mask_neg = document_pad_mask_neg.to(self.device)
            return query_emb, doc_pos_emb, doc_neg_emb, query_pad_mask, document_pad_mask_pos, document_pad_mask_neg
        else:
            _ = None
            query = batch["query_tokens"]
            doc_pos = batch["doc_tokens"]

            query = torch.nn.functional.pad(query["tokens"]["tokens"], (0, self.query_max - query["tokens"]["tokens"].size(1)), value=0)
            doc_pos = torch.nn.functional.pad(doc_pos["tokens"]["tokens"], (0, self.doc_max - doc_pos["tokens"]["tokens"].size(1)), value=0)

            query_pad_mask = (query > 1).float() # > 1 to also mask oov terms
            document_pad_mask_pos = (doc_pos > 1).float()

            query = {"tokens": {"tokens": query}}
            doc_pos = {"tokens": {"tokens": doc_pos}}
            
            query_emb = self.word_embedder(query).to(self.device)
            doc_pos_emb = self.word_embedder(doc_pos).to(self.device)
            query_pad_mask = query_pad_mask.to(self.device)
            document_pad_mask_pos = document_pad_mask_pos.to(self.device)
            return query_emb, doc_pos_emb, _, query_pad_mask, document_pad_mask_pos, _

        
class MovingAverageLoss:

    def __init__(self, window_size_batches=50):
        self.window_size_batches = window_size_batches
        self.losses_batches = []

    def add_loss(self, loss):
        self.losses_batches.append(loss)
    
    def get_window_loss(self):
        if len(self.losses_batches) >= self.window_size_batches:
            return sum(self.losses_batches[-self.window_size_batches:]) / self.window_size_batches

def tk_training_loop(model, loader, optimizer, scheduler, loss_criterion, batch_embedder, device, max_iter, epochs=2, BATCH_SIZE=64):
    # max_triples = 100000 # limit the number of triples to train on
    print_each = 50 # print and average over the last 50 batches
    for epoch in range(epochs):
        model.train()
        loss_accumulator = MovingAverageLoss(window_size_batches = print_each)
        for idx, batch in enumerate(loader):
            optimizer.zero_grad()
            query_emb, doc_pos_emb, doc_neg_emb, query_pad_mask, document_pad_mask_pos, document_pad_mask_neg = batch_embedder(batch)
            pred_pos = model(query_emb, doc_pos_emb, query_pad_mask, document_pad_mask_pos)
            pred_neg = model(query_emb, doc_neg_emb, query_pad_mask, document_pad_mask_neg)
            # clear memory
            del query_emb, doc_pos_emb, doc_neg_emb, query_pad_mask, document_pad_mask_pos, document_pad_mask_neg
            loss = loss_criterion(pred_pos, pred_neg, torch.ones(pred_pos.shape[0]).to(device))
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_accumulator.add_loss(loss.item())
            if (idx) % print_each == 0:
                print(f"Epoch: {epoch}, Batches: {idx}, Total triples: {idx * BATCH_SIZE}/{max_iter}, Average Loss: {loss_accumulator.get_window_loss()}, Current loss: {scheduler.get_last_lr()}")
            if max_iter and idx * BATCH_SIZE > max_iter:
                break
    return model, loss_accumulator