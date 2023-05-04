## Binary Question Answering with PyTorch

This project is an implementation of binary question answering using PyTorch, MLflow for model versioning, and Docker for containerization. 
The objective of this project is to create a binary question answering model that can answer yes/no questions based on the given question and a context text.



### Training dataset

The model is trained on the BoolQ dataset. The dataset is publicly available for download from [here](https://github.com/google-research-datasets/boolean-questions).

### Running the project

To run the project in a Docker container, follow the steps below:

Build the Docker image using the following command:

```bash
$ docker build -t binary-qa-pytorch-mlflow .
```

Start the Docker container using the following command:

```bash
$ docker run -it -p 5000:5000 binary-qa-pytorch-mlflow
```

The container will start the MLflow server and run the training script.
The logged model can be viewed in the MLflow UI by navigating to http://localhost:5000
