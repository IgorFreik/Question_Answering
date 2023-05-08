import yaml

with open('/config/config.yml', 'r') as f:
    config = yaml.safe_load(f)

model_config = config['bidaf']
model_params = model_config['model_params']
learning_params = model_config['model_learning']
mlflow_config = config['mlflow']
general = config['general']
