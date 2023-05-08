from tqdm import tqdm
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.model.bidaf import BiDAF


def get_xy(df_data, tokenizer, model, device):
    features = []
    labels = []

    with torch.no_grad():
        for _, row in tqdm(df_data.iterrows()):
            tokenized_passage = tokenizer.encode_plus(row['passage'], return_tensors='pt')
            tokenized_question = tokenizer.encode_plus(row['question'], return_tensors='pt')

            p_hidden = model(tokenized_passage['input_ids'].to(device),
                             tokenized_passage['attention_mask'].to(device)).hidden_states
            q_hidden = model(tokenized_question['input_ids'].to(device),
                             tokenized_question['attention_mask'].to(device)).hidden_states

            p_embeddings = torch.stack(p_hidden, dim=0).squeeze(1).cpu()
            q_embeddings = torch.stack(q_hidden, dim=0).squeeze(1).cpu()

            p_embeddings = p_embeddings[-4:]
            q_embeddings = q_embeddings[-4:]

            p_cls = p_embeddings[:, 0].reshape(-1)
            q_cls = q_embeddings[:, 0].reshape(-1)

            features.append(torch.cat([p_cls, q_cls], dim=0))
            labels.append(row['answer'])

    features = torch.stack(features).numpy()
    labels = np.array(labels).astype(int)
    return features, labels


def get_logistic_regression_score(model, tokenizer, train_data, test_data, device):

    train_features, train_labels = get_xy(train_data, tokenizer, model, device)
    test_features, test_labels = get_xy(test_data, tokenizer, model, device)

    lr_clf = LogisticRegression(max_iter=1000)
    lr_clf.fit(train_features, train_labels)
    return lr_clf.score(test_features, test_labels)


def predict(best_model, test_loader, device):

    pred_labels = []
    true_labels = []

    best_model.eval()
    with torch.no_grad():
        for i, batch in tqdm(enumerate(test_loader)):
            inputs = batch['inputs'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            output = best_model(inputs, attention_mask).cpu().numpy()

            true_labels.append(labels.numpy())
            pred_labels.append(output)

    true_labels = np.concatenate(true_labels, axis=0)
    pred_labels = np.concatenate(pred_labels, axis=0)
    return accuracy_score(true_labels, pred_labels)
