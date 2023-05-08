from torch.utils.data import Sampler, random_split
import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from src.data.dataset import QADataset


def get_padded(values):
    max_len = 0
    for value in values:
        if len(value) > max_len:
            max_len = len(value)

    padded = np.array([value + [0] * (max_len - len(value)) for value in values])

    return padded


def collate_fn(batch):
    word_contexts = []
    word_questions = []

    char_contexts = []
    char_questions = []

    labels = []

    for elem in batch:
        word_contexts.append(elem['word_context'])
        word_questions.append(elem['word_question'])

        char_contexts.append(elem['char_context'])
        char_questions.append(elem['char_question'])

        labels.append(elem['label'])

    word_contexts = get_padded(word_contexts)
    word_questions = get_padded(word_questions)

    char_contexts = get_padded(char_contexts)
    char_questions = get_padded(char_questions)

    return {"word_contexts": torch.tensor(word_contexts),
            "word_questions": torch.tensor(word_questions),
            "char_contexts": torch.tensor(char_contexts),
            "char_questions": torch.tensor(char_questions),
            'labels': torch.FloatTensor(labels)}


class QASampler(Sampler):
    def __init__(self, subset, batch_size=32):
        self.batch_size = batch_size
        # self.subset = subset

        # self.indices = subset.indices
        # # tokenized for our data
        self.word_contexts_tokenized = np.array(subset.dataset.word_contexts_tokenized)[subset.indices]
        self.word_questions_tokenized = np.array(subset.dataset.word_questions_tokenized)[subset.indices]

        self.char_contexts_tokenized = np.array(subset.dataset.char_contexts_tokenized)[subset.indices]
        self.char_questions_tokenized = np.array(subset.dataset.char_questions_tokenized)[subset.indices]

    def __iter__(self):

        batch_idx = []
        # index in sorted data
        for index in np.argsort(list(map(len, self.word_contexts_tokenized))):
            batch_idx.append(index)
            if len(batch_idx) == self.batch_size:
                yield batch_idx
                batch_idx = []

        if len(batch_idx) > 0:
            yield batch_idx

    def __len__(self):
        return len(self.dataset)


def get_loaders(dataset):
    train_size, val_size = int(.8 * len(dataset)), int(.1 * len(dataset))
    train_data, valid_data, test_data = random_split(dataset, [train_size, val_size, len(dataset) - train_size - val_size])

    train_loader = DataLoader(train_data, batch_sampler=QASampler(train_data), collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_sampler=QASampler(valid_data), collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_sampler=QASampler(test_data), collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader


def get_dataset(args):

    data = load_dataset('boolq')
    dataset = QADataset(args, data)

    return dataset
