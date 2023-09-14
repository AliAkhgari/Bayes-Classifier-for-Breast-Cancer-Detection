import numpy as np
import pandas as pd


class GaussianOptimalBayse:
    def __init__(self) -> None:
        self.theta = {}

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        for c in np.unique(y):
            x_c = x[y == c].to_numpy()
            self.theta[c] = {}
            self.theta[c]["mu"] = np.mean(x_c.T, axis=1)
            self.theta[c]["sigma"] = np.cov(x_c.T)
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
        p *= self.p_x_given_y(x=x, mu=self.theta[c]["mu"], sigma=self.theta[c]["sigma"])
        return p * self.theta[c]["p_y"]

    def p_x_given_y(self, x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
        d = x.shape[0]
        exp = -0.5 * (np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), (x - mu)))

        return (
            1 / ((2 * np.pi) ** (d / 2) * np.linalg.det(sigma) ** 1 / 2) * np.e**exp
        )
