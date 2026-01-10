import pandas as pd
import Functions as f
import acp.ACP as acp
import Graphics as g
import scipy.cluster.hierarchy as hclust
import scipy.spatial.distance as hdist
import sklearn.decomposition as dec
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

tabel = pd.read_csv('./dataIN/Life-Expectancy.csv', index_col=0)
print("Tabelul cu valori: " + "\n" + "-"*50)
print(tabel)

n = tabel.index.size
print('Numar observatii:', n, type(tabel.index))
obs = tabel.index.values
print(obs, type(obs))

m = tabel.columns[1:].size
print('Numar variabile:', m)
vars = tabel.columns[2:].values
print(vars, type(vars))

# standardizare matrice variabile observate X
X = tabel[vars].values

modelACP = acp.ACP(X)
X_std = modelACP.getXstd()
print(X_std, type(X_std), X_std.shape)

# Salvare X_std in CSV - optimizat
X_std_df = pd.DataFrame(data=X_std, index=obs, columns=vars)
X_std_df = X_std_df.astype(float)
X_std_df.to_csv('./dataOUT/X_std.csv', float_format='%.6f')
print("X_std salvat cu succes!")

# extragere valori proprii
alpha = modelACP.getAlpha()
g.valori_proprii(valori=alpha)
plt.savefig('./dataOUT/valori_proprii.svg', format='svg', bbox_inches='tight')
print("Grafic valori proprii salvat!")

# componente principale
comp = modelACP.getComponente()
comp_df = pd.DataFrame(data=comp,
            index=obs,
            columns=['C'+str(j+1) for j in range(comp.shape[1])])
comp_df.to_csv('./dataOUT/ComponentePrincipale.csv', float_format='%.6f')
print("Componente principale salvate!")

g.intesitate_legaturi(R2=comp_df, titlu='Componente principale')
plt.savefig('./dataOUT/componente_principale.svg', format='svg', bbox_inches='tight')
print("Grafic componente principale salvat!")

# Factor loadings
Rxc = modelACP.getFactorLoadings()
Rxc_df = pd.DataFrame(data=Rxc,
        index=vars,
        columns=['C'+str(j+1) for j in range(Rxc.shape[1])])
Rxc_df.to_csv('./dataOUT/FactorLoadings.csv', float_format='%.6f')
print("Factor loadings salvate!")

g.corelograma(R2=Rxc_df)
plt.savefig('./dataOUT/corelograma_factor_loadings.svg', format='svg', bbox_inches='tight')
print("Corelograma factor loadings salvată!")

# Scoruri
scoruri = modelACP.getScoruri()
scoruri_df = pd.DataFrame(data=scoruri,
            index=obs,
            columns=['C'+str(j+1) for j in range(scoruri.shape[1])])
scoruri_df.to_csv('./dataOUT/Scoruri.csv', float_format='%.6f')
print("Scoruri salvate!")

g.intesitate_legaturi(R2=scoruri_df,
        titlu='Scoruri - componente principale standardizate',
        color='Blues')
plt.savefig('./dataOUT/scoruri.svg', format='svg', bbox_inches='tight')
print("Grafic scoruri salvat!")

# Comunalitati
comun = modelACP.getComunalitati()
comun_df = pd.DataFrame(data=comun,
        index=vars,
        columns=['C'+str(j+1) for j in range(comun.shape[1])])
comun_df.to_csv('./dataOUT/Comunalitati.csv', float_format='%.6f')
print("Comunalitati salvate!")

g.corelograma(R2=comun_df, titlu='Graficul comunalitatilor')
plt.savefig('./dataOUT/comunalitati.svg', format='svg', bbox_inches='tight')
print("Grafic comunalități salvat!")

# Cercul corelatiilor
g.cercul_corelatiilor(R2=Rxc_df)
plt.savefig('./dataOUT/cercul_corelatiilor.svg', format='svg', bbox_inches='tight')
print("Cercul corelațiilor salvat!")

# Clusterizare
metode = list(hclust._LINKAGE_METHODS)
print(metode, type(metode))

metrici = hdist._METRICS_NAMES
print(metrici, type(metrici))

# Clusterizare ierarhica observatii
h_1 = hclust.linkage(y=X_std, method='single', metric='cityblock')
print(h_1)
threshold, j, k = f.threshold(h_1)
print(f"Threshold: {threshold}, j: {j}, k: {k}")

g.dendrogram(h=h_1, labels=obs,
             title="Clasificare ierarhica observatii (method='single', metric='cityblock')",
             threshold=threshold, colors=None)
plt.savefig('./dataOUT/dendrogram_observatii.svg', format='svg', bbox_inches='tight')
print("Dendrogramă observații salvată!")

# Clusterizare ierarhica variabile
h_2 = hclust.linkage(y=X_std.T, method='complete', metric='correlation')
print(h_2)
threshold, j, k = f.threshold(h_2)
print(f"Threshold: {threshold}, j: {j}, k: {k}")

g.dendrogram(h=h_2, labels=vars,
             title="Clasificare ierarhica variabile (method='complete', metric='correlation')",
             threshold=threshold, colors=None)
plt.savefig('./dataOUT/dendrogram_variabile.svg', format='svg', bbox_inches='tight')
print("Dendrogramă variabile salvată!")

g.afisare()

print("\n" + "="*50)
print("Toate graficele au fost salvate în ./dataOUT/ ca fișiere SVG!")
print("="*50)
