'''
Clasa care incapsuleaza implementarea modelului de ACP
'''
import numpy as np
import Functions as f



class ACP:
    def __init__(self, X):
        # asumam ca primim un numpy.ndarray in X
        self.X_std = f.standardizare(X)
        # medii = np.mean(a=X, axis=0)  # medii pe coloane
        # # axele se numeroteaza de la dreapta la stanga
        # abateri_std = np.std(a=X, axis=0, dtype=float)  # avem variabilele pe coloane
        # self.X_std = (X - medii) / abateri_std
        # cacul matrice de varianta-covaranta pentru X_std
        self.cov = np.cov(self.X_std, rowvar=False)
        # avem variabilele pe coloane
        # print(self.cov)
        valori, vectori = np.linalg.eigh(a=self.cov)
        print(valori, valori.shape)
        print(vectori.shape)
        # sortare descrescatoare a valorilor proprii si vectorilor proprii
        k_desc = [k for k in reversed(np.argsort(a=valori))]
        print(k_desc)
        self.alpha = valori[k_desc]
        self.a = vectori[:, k_desc]
        # regulazirea vectorilor proprii
        # inmultirea unui vector prorpiu cu un scalar nu modifica
        # natura vectorului propriu
        for j in range(self.a.shape[1]):
            min_col = np.min(a=self.a[:, j], axis=0) # calculam minim pe coloane
            max_col = np.max(a=self.a[:, j], axis=0) # calculam maxim pe coloane
            if np.abs(min_col) > np.abs(max_col):
                self.a[:, j] = -self.a[:, j]

        # calcul componente principale
        self.C = self.X_std @ self.a
        # calcul factor loadings
        self.Rxc = self.a * np.sqrt(self.alpha)


    def getXstd(self):
        return self.X_std

    def getAlpha(self):
        return self.alpha

    def getComponente(self):
        return self.C

    def getFactorLoadings(self):
        return self.Rxc

    def getScoruri(self):
        return self.C / np.sqrt(self.alpha)

    def getComunalitati(self):
        # Rxc2 = self.Rxc * self.Rxc
        Rxc2 = np.square(self.Rxc)
        return np.cumsum(a=Rxc2, axis=1) # sume cumulative pe linii


