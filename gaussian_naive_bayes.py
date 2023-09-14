import numpy as np
import pandas as pd


class GaussianNaiveBayse:
    def __init__(self) -> None:
        self.theta = {}

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        for c in np.unique(y):
            x_c = x[y == c].to_numpy()
            self.theta[c] = {}
            self.theta[c]["mu"] = np.mean(x_c.T, axis=1)
            self.theta[c]["std"] = np.std(x_c.T, axis=1)
            self.theta[c]["p_y"] = len(y[y == c]) / len(y)

    def predict(self, x: pd.DataFrame) -> pd.Series:
        index = x.index
        x = x.to_numpy()

        predicts = []
        for x_i in x:
            probs = []
            for c in self.theta.keys():
                p = self.p_y_given_x(x_i, c)
                probs.append(p)
            predicts.append(list(self.theta.keys())[np.argmax(probs)])

        return pd.Series(predicts, index=index)

    def p_y_given_x(self, x: np.ndarray, c: int) -> float:
        p = 1.0
        for i in range(len(x)):
            p *= self.p_x_given_y(
                x=x[i], mu=self.theta[c]["mu"][i], std=self.theta[c]["std"][i]
            )
        return p * self.theta[c]["p_y"]

    def p_x_given_y(self, x: float, mu: float, std: float) -> float:
        exp = -0.5 * ((x - mu) / std) ** 2
        return 1 / (np.sqrt(2 * np.pi) * std) * np.e**exp
