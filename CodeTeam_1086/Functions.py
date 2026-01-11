import numpy as np
import Graphics as graphics
import pandas as pd
import pandas.api.types as pdt


def standardizare(X):
    """
        Standardizează matricea X (z-score).
        Formula: z = (x - medie) / abatere_standard
        Rezultat: fiecare coloană are media=0 și std=1
    """
    medii = np.mean(a=X, axis=0) # medii pe coloane
    # Axele se numerotează de la dreapta la stânga
    abateri_std = np.std(a=X, axis=0) # Avem variabilele pe coloane
    return (X - medii) / abateri_std


def center(x):
    """
    Centrează datele (scade media).
    Rezultat: media fiecărei coloane devine 0.
    Diferență față de standardizare: NU împarte la std.
    """
    means = np.mean(x, axis=0)
    return (x - means)


def regularise(t, y=None):
    """
    Regularizează/normalizează datele într-un interval.
    t: tabelul de date
    y: valorile țintă (opțional)
    """

    if isinstance(t, pd.DataFrame):
        for c in t.columns:
            minim = t[c].min()
            maxim = t[c].max()
            if abs(minim) > abs(maxim):
                t[c] = -t[c]
                if y is not None:
                    k = t.columns.get_loc(c)
                    y[:, k] = -y[:, k]
    if isinstance(t, np.ndarray):
        for i in range(np.shape(t)[1]):
            minim = np.min(t[:, i])
            maxim = np.max(t[:, i])
            if np.abs(minim) > np.abs(maxim):
                t[:, i] = -t[:, i]
    return None



def replace_na_df(t):
    """
    Înlocuiește NaN în DataFrame pandas.
    Pentru coloane numerice: înlocuiește cu media.
    Pentru coloane categoriale: înlocuiește cu moda (cea mai frecventă valoare).
    """

    for c in t.columns:
        if pdt.is_numeric_dtype(t[c]):
            if t[c].isna().any():
                avg = t[c].mean()
                t[c] = t[c].fillna(avg)
        else:
            if t[c].isna().any():
                mode = t[c].mode()
                t[c] = t[c].fillna(mode[0])
    return None


def replace_na(X):
    # înlocuiește valorile lipsă (NaN) cu media coloanei
    means = np.nanmean(X, axis=0)
    k_nan = np.where(np.isnan(X))
    X[k_nan] = means[k_nan[1]]
    return X


def cluster_distribution(h, k):
    # Calculează distribuția observațiilor în k clustere.
    # h - este ierarhia de clustere numpy.ndarray
    # k - numar clustere din partiția de maximă stabilitate
    n = np.shape(h)[0] + 1  # Numărul de clustere de pe primul nivel al ierarhiei
    g = np.arange(0, n)
    print(g)
    for i in range(n-k):  # Iteram până la joncțiune ce dă partiția de maximă stabilitate
        k1 = h[i, 0]
        k2 = h[i, 1]
        g[g==k1] = n + i
        g[g==k2] = n + i
    print(g)
    g_ = pd.Categorical(values=g)
    print(g_)
    return ['C'+str(i+1) for i in g_.codes], g_.codes


def threshold(h):
    # Găsește pragul optim = saltul maxim în distanțele de agregare

    m = np.shape(h)[0]  # Numărul maxim de joncțiuni
    dist_1 = h[1:, 2]
    dist_2 = h[:m-1, 2]
    dif = dist_1 - dist_2
    print(dif)
    j = np.argmax(dif)  # joncțiunea unde avem partiția de maximă stabilitate
    print(j)
    threshold = (h[j, 2] + h[j+1, 2]) / 2
    return threshold, j, m



def cluster_display(g, row_labels, col_label, file_name):
    pairs = zip(row_labels, g)
    pairs_list = [g for g in pairs]
    print(pairs_list)
    g_dict = {k: v for (k, v) in pairs_list}
    print(g_dict)
    g_df = pd.DataFrame.from_dict(data=g_dict,
                orient='index', columns=[col_label])
    print(g_df)
    # salvare grupare in in fisier CSV
    g_df.to_csv('./dataOUT/' + file_name)


def color_clusters(h, k, codes):
    # atribuie culori clusterelor și joncțiunilor

    colors = np.array(graphics._COLORS)
    nr_colors = len(colors)
    m = np.shape(h)[0]
    n = m + 1
    cluster_colors = np.full(shape=(2 * n * 1,),
                             fill_value="", dtype=np.chararray)
    for i in range(n):
        cluster_colors[i] = colors[codes[i] % nr_colors]
    for i in range(m):
        k1 = int(h[i, 0])
        k2 = int(h[i, 1])
        if cluster_colors[k1] == cluster_colors[k2]:
            cluster_colors[n + i] = cluster_colors[k1]
        else:
            cluster_colors[n + i] = 'k'
    return cluster_colors
