import random
import csv
from naive_bayes import train_NBC, predict_NBC

def load_csv(filename, dtype=float):
    with open(filename, "r", newline="") as file:
        reader = csv.reader(file)
        return [
            [dtype(value) for value in row]
            for row in reader
            if row
        ]

def calculate_accuracy(predictions, Y_test):
    correct = 0

    for predicted, actual in zip(predictions, Y_test):
        predicted_class = predicted[0]
        actual_class = actual[0] if isinstance(actual, (list, tuple)) else actual

        if predicted_class == actual_class:
            correct += 1

    return correct / len(Y_test)


def evaluate_NBC(X, Y, X_dtype, L, D_categorical, repetitions=100):
    training_percentages = [10, 40, 60, 75, 90]
    results = {}

    for percentage in training_percentages:
        accuracies = []

        for repetition in range(repetitions):
            indices = list(range(len(X)))
            random.shuffle(indices)

            train_size = int(len(X) * percentage / 100)

            train_indices = indices[:train_size]
            test_indices = indices[train_size:]

            X_train = [X[i] for i in train_indices]
            Y_train = [Y[i] for i in train_indices]

            X_test = [X[i] for i in test_indices]
            Y_test = [Y[i] for i in test_indices]

            model = train_NBC(
                X_train,
                X_dtype,
                Y_train,
                L,
                D_categorical
            )

            predictions = predict_NBC(
                model,
                X_test,
                X_dtype
            )

            accuracy = calculate_accuracy(predictions, Y_test)
            accuracies.append(accuracy)

        average_accuracy = sum(accuracies) / len(accuracies)
        results[percentage] = average_accuracy

    return results


if __name__ == "__main__":
    # Dataset A: Categorical
    X_A = load_csv("datasets/DatasetA_X_categorical.csv", int)
    Y_A = load_csv("datasets/DatasetA_Y.csv", int)

    D_A = load_csv("datasets/DatasetA_D_categorical.csv", int)
    D_A = [value for row in D_A for value in row]

    # Dataset B: Continuous
    X_B = load_csv("datasets/DatasetB_X_continuous.csv", float)
    Y_B = load_csv("datasets/DatasetB_Y.csv", int)

    print("Dataset A:", len(X_A), "samples,", len(X_A[0]), "features")
    print("Dataset B:", len(X_B), "samples,", len(X_B[0]), "features")
    print("Dataset A categories:", D_A)

    results_A = evaluate_NBC(X_A, Y_A, "categorical", 1, D_A)

    results_B = evaluate_NBC(X_B, Y_B, "continuous", 0, None)

    print("\nDataset A:")
    for percentage, accuracy in results_A.items():
        print(f"{percentage}% training: {accuracy:.2%}")

    print("\nDataset B:")
    for percentage, accuracy in results_B.items():
        print(f"{percentage}% training: {accuracy:.2%}")