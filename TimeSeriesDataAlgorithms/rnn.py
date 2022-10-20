import os
import sys
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
from torch.utils.data import DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.x = data
        self.y = labels

    def __len__(self):
        return len(labels)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def collate_fn(data):
    patient_data, labels = zip(*data)
    y = torch.tensor(labels, dtype = torch.float)
    num_patients = len(patient_data)
    num_visits = [len(patient) for patient in patient_data]
    num_codes = [len(visit) for patient in patient_data for visit in patient]

    max_num_visits = max(num_visits)
    max_num_codes = max(num_codes)

    x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype = torch.long)
    rev_x = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype = torch.long)
    masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype = torch.bool)
    rev_masks = torch.zeros((num_patients, max_num_visits, max_num_codes), dtype = torch.bool)

    for i_patient, patient in enumerate(patient_data):
        for j_visit, visit in enumerate(patient):
            total_patient = len(patient)
            total_visits = len(visit)

            x[i_patient, j_visit, : total_visits] = torch.tensor(visit, dtype = torch.long)
            rev_x[i_patient, total_patient - j_visit - 1, : total_visits] = torch.tensor(visit, dtype = torch.long)
            masks[i_patient, j_visit, :total_visits].fill_(1)
            rev_masks[i_patient, total_patient - j_visit - 1, :total_visits].fill_(1)

    return x, masks, rev_x, rev_masks, y


def load_data(train_dataset, val_dataset, collate_fn):
    batch_size = 32
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, collate_fn = collate_fn)
    val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, collate_fn = collate_fn)
    return train_loader, val_loader

def sum_embeddings_with_mask(x, masks):
    masks = masks.unsqueeze(-1).float()
    embeds = x * masks
    sum_embeddings = embeds.sum(dim = 2)
    return sum_embeddings


def get_last_visit(hidden_states, masks):
    masks = torch.sum(masks, dim = 2) > 0
    last_state = (torch.sum(masks, dim = 1) -1).view(-1,1,1).expand(-1, -1, hidden_states.shape[2])
    last_hidden_state = torch.squeeze(torch.gather(hidden_states, dim = 1, index = last_state), 1)
    return last_hidden_state


class RNN(nn.Module):
    def __init__(self, num_codes):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings = num_codes, embedding_dim = 128)
        self.rnn = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True)
        self.rev_rnn = nn.GRU(input_size = 128, hidden_size = 128, batch_first = True, bidirectional = True)
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, masks, rev_x, rev_masks):
        batch_size = x.shape[0]
        x = self.embedding(x)
        x = sum_embeddings_with_mask(x, masks)
        output, hidden = self.rnn(x)
        true_h_n = get_last_visit(output, masks)
        rev_x = sum_embeddings_with_mask(self.embedding(rev_x), rev_masks)
        output, hidden = self.rnn(rev_x)
        true_h_n_rev = get_last_visit(output, rev_masks)
        logits = self.fc(torch.cat([true_h_n, true_h_n_rev], 1))
        prob = self.sigmoid(logits)
        return prob.view(batch_size)


def eval_model(model, val_loader):
    model.eval()
    y_pred = torch.LongTensor()
    y_score = torch.Tensor()
    y_true = torch.LongTensor()
    for x, masks, rev_x, rev_masks, y in val_loader:
        y_hat = model(x, masks, rev_x, rev_masks)
        y_score = torch.cat((y_score, y_hat.detach().to('cpu')), dim = 0)
        y_hat = (y_hat > 0.5).int()
        y_pred = torch.cat((y_pred, y_hat.detach().to('cpu')), dim = 0)
        y_true = torch.cat((y_true, y.detach().to('cpu')), dim = 0)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average = "binary")
    roc_auc = roc_auc_score(y_true, y_score)
    return p, r, f, roc_auc


def train_model(model, train_loader, val_loader, n_epochs):
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for x, masks, rev_x, rev_masks, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x, masks, rev_x, rev_masks)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        print('Epoch: {} \t Training Loss: {:.6f}'.format(epoch +1, train_loss))
        p, r, f, roc_auc = eval_model(model, val_loader)
        print('Epoch: {} \t Validation p: {:.2f}, r: {:.2f}, f: {:.2f}, roc_auc: {:.2f}'.format(epoch + 1, p, r, f, roc_auc))


patient_data = pickle.load(open('./patient_visit_icds', 'rb'))
labels = pickle.load(open('./labels', 'rb'))

num_codes = 5056
dataset = CustomDataset(patient_data, labels)
split = int(len(dataset) * 0.8)
lengths = [split, len(dataset) - split]
train_dataset, val_dataset = random_split(dataset, lengths)
print("Length of training dataset: ", len(train_dataset))
print("Length of validation dataset: ", len(val_dataset))

train_loader, val_loader = load_data(train_dataset, val_dataset, collate_fn)

rnn = RNN(num_codes = 5056)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr = 0.001)

n_epochs = 10
train_model(rnn, train_loader, val_loader, n_epochs)
