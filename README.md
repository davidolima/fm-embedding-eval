# Foundation Models Embedding Evaluation

This repository was made for experimenting with the embeddings produced by different histopathology foundation models. 
They were evaluated over the task of glomerulus lesion classification.

# Setup
 - Create a Python Virtual Environment and install required packages.
```sh
python3 -m venv .
./bin/python3 -m pip install -r requirements.txt
```

# Replicating the results
For more options or more specific help, use the flag `--help` before running any given script.

## Feature extraction
```sh
./bin/python3 -m embedding.extraction\
     -i /path/to/dataset/ \
     -o /output/path/ \
     --classes class1 class2 class3 ...
```

## EfficientNet-B0 training
```sh
./bin/python3 -m scripts.train_effnetb0 
```

## Generating graphs
```sh
./bin/python3 -m scripts.cross-validation_performance_graph /path/to/generated/logs/folder/ 
```

