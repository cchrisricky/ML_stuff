import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    df = pd.read_csv("QM9.csv")
    energies = df["E (Hartree)"].to_numpy()

ave_E = np.mean(energies)
std_E = np.std(energies)

print("average E:", ave_E, "Hartree")
print("or", ave_E* 2625.5, "kJ/mol")
print("std of E:", std_E, "Hartree")
print("or", std_E * 2625.5, "kJ/mol")
plt.hist(energies, bins = 100)
plt.show()
