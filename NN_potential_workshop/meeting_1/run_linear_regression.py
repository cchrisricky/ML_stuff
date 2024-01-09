import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

if __name__ == "__main__":

    df = pd.read_csv("QM9.csv")
    
    X = df[["n_H", "n_C", "n_N", "n_O", "n_F"]].to_numpy()
    y = df[["E (Hartree)"]].to_numpy()
    
    model = LinearRegression(fit_intercept=True)
    
    reg = model.fit( X, y )

    coef = reg.coef_
    intercept = reg.intercept_

    y_hat = reg.predict(X)

    RMSE = np.sqrt(np.mean((y_hat - y)**2))
    MAE = np.mean(np.absolute(y_hat - y))

    print("root mean square error:", RMSE, "Hartree or", RMSE * 2625.5, "kJ/mol")
    print("mean absolute error:", MAE, "Hartree or", MAE * 2625.5, "kJ/mol")
    
    print("weights for H, C, N, O, F:", coef[0,0], coef[0,1], coef[0,2], coef[0,3], coef[0,4])
    print("bias =", intercept[0], "Hartree")

    fig, ax = plt.subplots()
    ax.scatter(y, y_hat)
    ax.axline((0, 0), slope = 1, color = 'black', lw = 2)
    plt.savefig("yhat-y_scatter.png", dpi=300)
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(y, y_hat - y)
    ax.axline((0, 0), slope = 0, color = 'black', lw = 2)
    plt.savefig("delY-Y_scatter.png", dpi=300)
    plt.show()

