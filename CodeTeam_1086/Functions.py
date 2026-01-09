import numpy as np


def standardizare(X):
    # asumam ca primim ca paramentru
    # un numpy.ndarray
    medii = np.mean(a=X, axis=0) # medii pe coloane
    # axele se numeroteaza de la dreapta la stanga
    # print(medii.shape)
    abateri_std = np.std(a=X, axis=0) # avem variabilele pe coloane
    return (X - medii) / abateri_std

# obligatoriu in acp
