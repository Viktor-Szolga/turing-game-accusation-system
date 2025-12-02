# Reverse Turing Game Classifier – Bachelor Thesis
This repository contains the code, trained models, and experiment configurations used for my bachelor thesis. It provides all scripts and data required to recreate the experimental results and plots described in the thesis.

## Installation
The project uses [Poetry](https://python-poetry.org/) for dependency management.

1. Clone this repository:
   ```bash
   git clone https://github.com/Viktor-Szolga/turing-game-accusation-system.git
   cd turing-game-accusation-system
   ```

2. Create the environment with Poetry:
   ```bash
   pip install .
   ```

   Alternatively, you can install from the frozen requirements file:
   ```bash
   pip install -r requirements_freeze.txt
   ```

## Running Experiments
The entry point to train models is:
```bash
python train.py
```
This script:
- Trains all classifier configurations from the specified folder.
- Stores results and model checkpoints in the appropriate folders.


To evaluate the final model on the test set you can use:
```bash
python test.py
```

All experiment configurations are stored as YAML files in the `experiments/` folder.

## Trained Models
- All trained model state dictionaries are saved in the `trained_models/` directory.
- The final model selected in the thesis can be found in:
  `final_model/scaled_model_d.pth`

## Data and Preprocessing
- **BoW Encodings:** To calculate the Bag of Words encodings used run:
```bash
python data/generate_encodings/BoW_encodings.py
```
- **Sentence Embeddings:** Precomputed encodings are included in the repository. (They were originally generated via the `encode_data.ipynb` notebook, which connects to a local server and therefore cannot be rerun directly.)

## Recreating Plots
Use the different files in `scripts/plots` to recreate the plots from the thesis.

## Analysis and Evaluation
The repository also includes notebooks with the analysis and evaluation performed during the thesis:

- `analysis/BoW_vs_Sentence_Embedding_KNN.ipynb`: Contains the training of the KNN models presented in the thesis.
- `analysis/data_analysis.ipynb`: Contains the data exploration and analysis steps.
- `analysis/classifier_evaluation.ipynb`: Contains the evaluation of the trained classifiers.

The analysis in classifier_evaluation requires the encoding model running on the Turing Game server and can thus also not be rerun directly.
