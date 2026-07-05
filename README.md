# Requirements

- `python 3.14`
- `uv >=0.11.26 <0.12`

Setup a `.env` file at the top folder with an OpenAI key that allows for `gpt-5.4-mini` and `gpt-5.4-nano`

# How to run the project

## Test the model

`uv run --env-file .env --with jupyter jupyter lab`

This will launch a jupyter lab interface that will allow you to run the model in the `signal_generation.ipynb` file.

It is of note that the import on the first cell points to a .pkl that contains the model, so changing it if you build your own model might be necessary.

## Auxiliary Scripts

### 1. Dataset Generator

`uv run --env-file .env python -m hospitality.scripts.dataset_generator`

### 2. Embeddings Generator

`uv run --env-file .env python -m hospitality.scripts.embeddings_generator`

### 3. Feature Extractor (LLM Generated Feature generation)

`uv run --env-file .env python -m hospitality.scripts.feature_extractor`

### 4. Feature Processor (Computationally Generated Feature generation)

`uv run --env-file .env python -m hospitality.scripts.dataset_generator`

### 5. Model training

`uv run --env-file .env --with jupyter jupyter lab`

This will launch a jupyter lab interface that will allow you to run the model in the `scripts/lightgbm_training.ipynb` file.