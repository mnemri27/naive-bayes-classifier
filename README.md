# Naive Bayes Classifier from Scratch

A simple implementation of the Naive Bayes classifier in Python for my Machine Learning (CS-577) assignment at the University of Crete.

The project supports two types of data:
- **Categorical:** Uses conditional probabilities with Laplace smoothing.
- **Continuous:** Uses Gaussian distributions based on the mean and variance of each feature.

The classifier is implemented without NumPy, scikit-learn, or other machine learning libraries.

## Files

- `naive_bayes.py` — Contains `train_NBC()` and `predict_NBC()` for training and prediction.
- `evaluate.py` — Evaluates the classifier using different training and testing splits.
- `datasets/` — Contains the provided CSV datasets.

## Evaluation

The classifier was tested on two datasets using training sizes of 10%, 40%, 60%, 75%, and 90%. Each experiment was repeated 100 times, and the average accuracy was calculated.

| Training | Dataset A | Dataset B |
|---|---:|---:|
| 10% | 81.90% | 88.61% |
| 40% | 84.83% | 93.80% |
| 60% | 85.31% | 94.48% |
| 75% | 85.52% | 94.52% |
| 90% | 86.37% | 94.74% |

## Run

From the project directory:

```bash
python evaluate.py
```

Or with `uv`:

```bash
uv run python evaluate.py
```

The results may vary slightly between runs because the data is randomly shuffled.
