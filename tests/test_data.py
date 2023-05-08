from datasets import load_dataset
import numpy as np
import pytest
from src.data.dataset import QADataset
from src.data.get_data import QASampler, collate_fn, get_padded, get_loaders, get_dataset
from torch.utils.data import DataLoader, random_split


args = {
    'word_dim': 100
}


examples = ['This is a simple sentence.', 'Here is, another!', 'Those will be used for: testing']


def test_word_tokenizing():
    tokenized1 = QADataset.word_tokenizing(examples[0])
    tokenized2 = QADataset.word_tokenizing(examples[1])
    tokenized3 = QADataset.word_tokenizing(examples[2])

    assert tokenized1 == ['this', 'is', 'a', 'simple', 'sentence', '.']
    assert tokenized2 == ['here', 'is', ',', 'another', '!']
    assert tokenized3 == ['those', 'will', 'be', 'used', 'for', 'testing']


def test_dataset_getitem():
    pass


def test_get_padded():
    sentences = [[1, 2, 3, 4], [2, 3], [1, 3, 4]]
    padded = get_padded(sentences)
    assert padded == np.array([[1, 2, 3, 4], [2, 3, 0, 0], [1, 3, 4, 0]])


def test_get_dataset():
    dataset = get_dataset(args)
    assert isinstance(dataset, QADataset)


def test_sampler_init():
    dataset = get_dataset(args)
    train_size, val_size = int(.8 * len(dataset)), int(.1 * len(dataset))
    train_data, valid_data, test_data = random_split(dataset,
                                                     [train_size, val_size, len(dataset) - train_size - val_size])
    sampler = QASampler(train_data)
    len_data = 0
    for ids in sampler:
        len_data += len(ids)
    assert len_data == len(train_data)


def test_get_loaders():
    dataset = get_dataset(args)
    train_loader, val_loader, test_loader = get_loaders(dataset)
    batch = next(iter(train_loader))

    word_contexts = batch['word_contexts']
    word_questions = batch['word_questions']
    char_contexts = batch['char_contexts']
    char_questions = batch['char_questions']
    labels = batch['labels']

    assert len(word_questions) == len(labels)
    assert len(char_questions) == len(labels)
    assert len(word_contexts) == len(labels)
    assert len(char_contexts) == len(labels)


def test_get_pretrained_emb():
    dataset = get_dataset(args)
    embeddings = dataset.get_pretrained_emb()
    assert len(embeddings.shape) == 2
    assert embeddings.shape[1] == args['word_dim']
