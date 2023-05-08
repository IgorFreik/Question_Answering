from torch.utils.data import Dataset
from torchtext.vocab import GloVe, vocab
from torchtext.data.utils import get_tokenizer
from collections import Counter, OrderedDict
import torch


class QADataset(Dataset):
    def __init__(self, args, subset):

        self.glove_vectors = GloVe(name='6B', dim=args.word_dim)
        self.glove_vocab = vocab(self.glove_vectors.stoi, specials=['<unk>'])
        self.glove_vocab.set_default_index(self.glove_vocab['<unk>'])

        self.char_vocab = self.build_char_vocab(subset['passage'], subset['question'])

        self.answers = subset['answer']
        self.word_contexts_tokenized = subset['passage'].apply((lambda x: self.glove_vocab(self.word_tokenizing(x))))
        self.word_questions_tokenized = subset['question'].apply((lambda x: self.glove_vocab(self.word_tokenizing(x))))

        self.char_contexts_tokenized = subset['passage'].apply((lambda x: self.char_tokenizing(self.word_tokenizing(x))))
        self.char_questions_tokenized = subset['question'].apply((lambda x: self.char_tokenizing(self.word_tokenizing(x))))

    def __getitem__(self, idx):
        return {"word_context": self.word_contexts_tokenized[idx],
                "word_question": self.word_questions_tokenized[idx],
                "char_context": self.char_contexts_tokenized[idx],
                "char_question": self.char_questions_tokenized[idx],
                "label": self.answers[idx]}

    def char_tokenizing(self, tok_seqs):
        return [[self.char_vocab[letter] for letter in tok] for tok in tok_seqs]

    def word_tokenizing(self, sent):
        return [token.replace("''", '"').replace("``", '"').replace('ˈ', "'").replace("`", "'") for token in
                get_tokenizer("basic_english")(sent)]

    def get_pretrained_emb(self):
        pretrained_embeddings = self.glove_vectors.vectors
        pretrained_embeddings = torch.cat((torch.zeros(1, pretrained_embeddings.shape[1]), pretrained_embeddings))
        return pretrained_embeddings

    def build_char_vocab(self, text_c, text_q):
        counter = Counter()
        for sentence in text_c:
            counter.update(sentence)
        for sentence in text_q:
            counter.update(sentence)

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        char_vocab = vocab(ordered_dict, min_freq=10, specials=['<unk>'])
        char_vocab.set_default_index(char_vocab['<unk>'])
        return char_vocab

    def get_char_vocab_len(self):
        return len(self.char_vocab)

    def __len__(self):
        return len(self.answers)
