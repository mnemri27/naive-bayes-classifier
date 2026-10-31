import math


def train_NBC(X, X_dtype, Y, L, D_categorical):
    if not X or not Y:
        raise ValueError("Training data cannot be empty")

    if len(X) != len(Y):
        raise ValueError("X and Y must have the same number of samples")

    if X_dtype not in ("categorical", "continuous"):
        raise ValueError("Unsupported feature type")

    if L < 0:
        raise ValueError("L must be non-negative")

    if any(len(row) != len(X[0]) for row in X):
        raise ValueError("All samples must have the same number of features")

    Y = [y[0] if isinstance(y, (list, tuple)) else y for y in Y]

    n_samples = len(Y)
    n_features = len(X[0])

    # Count samples in each class
    class_counts = {}

    for y in Y:
        if y not in class_counts:
            class_counts[y] = 0
        class_counts[y] += 1

    # Calculate class prior probabilities
    priors = {}

    for c in class_counts:
        priors[c] = class_counts[c] / n_samples

    model = {
        "classes": list(class_counts.keys()),
        "priors": priors,
        "X_dtype": X_dtype
    }

    # Calculate categorical conditional probabilities
    if X_dtype == "categorical":
        probabilities = {}

        for c in class_counts:
            probabilities[c] = []

            # Select the samples belonging to class c
            X_class = [
                X[i] for i in range(n_samples)
                if Y[i] == c
            ]

            for m in range(n_features):
                D = D_categorical[m]
                feature_probabilities = []

                for value in range(D):
                    count = 0

                    for sample in X_class:
                        if sample[m] == value:
                            count += 1

                    probability = (count + L) / (
                        class_counts[c] + L * D
                    )

                    feature_probabilities.append(probability)

                probabilities[c].append(feature_probabilities)

        model["probabilities"] = probabilities
        model["D_categorical"] = D_categorical
    
    elif X_dtype == "continuous":
        means = {}
        variances = {}

        for c in class_counts:
            X_class = [
                X[i] for i in range(n_samples)
                if Y[i] == c
            ]

            means[c] = []
            variances[c] = []

            for m in range(n_features):
                values = [sample[m] for sample in X_class]

                mean = sum(values) / len(values)

                variance = sum(
                    (value - mean) ** 2
                    for value in values
                ) / len(values)

                means[c].append(mean)
                variances[c].append(variance)

        model["means"] = means
        model["variances"] = variances


    return model


def predict_NBC(model, X, X_dtype):
    if X_dtype != model["X_dtype"]:
        raise ValueError("Feature type does not match the model")

    predictions = []

    for sample in X:
        best_class = None
        best_score = float("-inf")

        for c in model["classes"]:
            score = math.log(model["priors"][c])

            for m, value in enumerate(sample):

                if X_dtype == "categorical":
                    probability = model["probabilities"][c][m][value]

                    if probability == 0:
                        score = float("-inf")
                        break

                    score += math.log(probability)

                elif X_dtype == "continuous":
                    mean = model["means"][c][m]
                    variance = max(model["variances"][c][m], 1e-9)

                    score += (
                        -0.5 * math.log(2 * math.pi * variance)
                        - ((value - mean) ** 2) / (2 * variance)
                    )

            if best_class is None or score > best_score:
                best_score = score
                best_class = c

        predictions.append([best_class])

    return predictions







