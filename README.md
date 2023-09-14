# Bayes-Classifier-for-Breast-Cancer-Detection

I have implemented `Gaussian Naive Bayes` and `Gaussian Optimal Bayes` entirely from scratch in this project. These algorithms are employed for the vital task of breast cancer detection.

## Gaussian Naive Bayes
Gaussian Naive Bayes (GNB) is a probabilistic classifier ideal for Gaussian-distributed data. It assumes **feature independence**, making it efficient for many tasks, but less suitable for correlated features.

## Gaussian Optimal Bayes
Gaussian Optimal Bayes (GOB) extends GNB by considering **feature covariance**, accommodating correlated features. It uses multidimensional Gaussian distributions, making it more flexible in capturing complex relationships. GOB is suitable when features are not independent but is computationally more demanding than GNB.

## Usage
```python
import pandas as pd
from gaussian_naive_bayes import GaussianNaiveBayse
from gaussian_optimal_bayes import GaussianOptimalBayse
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

df = pd.read_csv("Data/Breast_cancer_data.csv")

x = df[
    [
        "mean_radius",
        "mean_texture",
        "mean_perimeter",
        "mean_area",
        "mean_smoothness",
    ]
]
y = df["diagnosis"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1234
)

gob = GaussianOptimalBayse()  # gnb = GaussianNaiveBayse()
gob.fit(x=x_train, y=y_train)  # gnb.fit(x=x_train, y=y_train)
y_pred = gob.predict(x=x_test)  # y_pred = gnb.predict(x=x_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print("Confusion Matrix: {conf_matrix}")
