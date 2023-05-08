import math
import mlflow
import mlflow.pytorch
import time
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer

from config.config import model_config, model_params, learning_params, mlflow_config
from src.data.get_data import get_loaders, get_dataset
from src.model.predict import predict
from src.model.bidaf import BiDAF
from src.model.train import train, evaluate
from src.utils.utils import epoch_time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    # Set experiment name
    mlflow.set_tracking_url(mlflow_config['tracking_url'])
    experiment = mlflow.get_experiment_by_name(model_config['model_name'])
    if experiment is None:
        experiment_id = mlflow.create_experiment(mlflow_config['experiment_name'])
    else:
        experiment_id = experiment.experiment_id

    # Loading data
    dataset = get_dataset(model_config)
    train_loader, valid_loader, test_loader = get_loaders(dataset)

    # Creating model
    if model_config['model_name'] == 'BiDAF':
        model = BiDAF(model_params, dataset.get_char_vocab_len(), dataset.get_pretrained_emb())
    elif model_config['model_name'] == 'BERT':
        model = AutoModel('bert-base-uncased').to(device)
        tokenizer = AutoTokenizer('bert-base-uncased')
    else:
        raise 'KeyModelNameError'

    # Set config values
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training
    train_history = []
    valid_history = []

    N_EPOCHS = 3
    CLIP = 1

    best_valid_loss = float('inf')
    with mlflow.start_run(experiment_id=experiment_id) as run:
        for epoch in range(N_EPOCHS):

            start_time = time.time()

            train_loss = train(model, train_loader, optimizer, criterion, CLIP, train_history, valid_history)
            valid_loss = evaluate(model, valid_loader, criterion)

            end_time = time.time()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best-model.pt')

            train_history.append(train_loss)
            valid_history.append(valid_loss)
            print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

        best_model = BiDAF(model_params, dataset.get_pretrained_emb()).to(device)
        best_model.load_state_dict(torch.load('best-model.pt'))
        mlflow.pytorch.log_model(best_model, 'model')

        best_accuracy = predict(best_model, test_loader, device)
        mlflow.log_metric('accuracy', best_accuracy)


if __name__ == '__main__':
    main()
