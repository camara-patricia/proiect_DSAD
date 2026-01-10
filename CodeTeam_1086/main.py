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

vars = tabel.columns[2:].values
m = len(vars)
print('Numar variabile:', m)
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

# =========================================================
# (NOU) Varianță explicată + cumulată (ACP) + alegere nr. componente
# =========================================================
alpha = np.array(alpha).astype(float)
var_exp = alpha / np.sum(alpha) * 100.0
cum_var_exp = np.cumsum(var_exp)

k_labels = ['C' + str(i+1) for i in range(len(alpha))]
var_exp_df = pd.DataFrame({
    "Valoare_proprie": alpha,
    "Var_exp_%": var_exp,
    "Var_exp_cumulata_%": cum_var_exp
}, index=k_labels)

var_exp_df.to_csv("./dataOUT/VarExplicata_ACP.csv", float_format="%.6f")
print("Varianța explicată salvată în ./dataOUT/VarExplicata_ACP.csv")

# Criteriul Kaiser (alpha > 1)
s_kaiser = int(np.sum(alpha > 1.0))
# asigură minim 2 pentru plan (C1, C2)
if s_kaiser < 2:
    s_kaiser = 2

print(f"Număr componente după Kaiser (alpha>1): {s_kaiser}")
print(f"Varianță cumulată primele {s_kaiser} componente: {cum_var_exp[s_kaiser-1]:.2f}%")

# =========================================================
# (NOU) Cos² (calitatea reprezentării) - observații
# =========================================================
# scoruri_df există deja în codul tău (Scoruri.csv)
F = scoruri_df.values.astype(float)  # n x p

dist2 = np.sum(F**2, axis=1, keepdims=True)  # n x 1
# evită împărțire la 0 (rar)
dist2[dist2 == 0] = 1e-12

cos2 = (F**2) / dist2  # n x p
cos2_df = pd.DataFrame(cos2, index=scoruri_df.index, columns=scoruri_df.columns)
cos2_df.to_csv("./dataOUT/Cos2_Observatii.csv", float_format="%.6f")
print("Cos² observații salvat în ./dataOUT/Cos2_Observatii.csv")

# Calitate cumulată pe primele 2 componente (C1 + C2)
if "C1" in cos2_df.columns and "C2" in cos2_df.columns:
    cos2_df["Calitate_C1_C2"] = cos2_df["C1"] + cos2_df["C2"]
    cos2_df[["Calitate_C1_C2"]].to_csv("./dataOUT/Calitate_C1_C2.csv", float_format="%.6f")
    print("Calitate C1+C2 salvată în ./dataOUT/Calitate_C1_C2.csv")


# =========================================================
# (NOU) Contribuția observațiilor la axe (CTR obs)
# =========================================================
sum_sq_per_axis = np.sum(F**2, axis=0, keepdims=True)  # 1 x p
sum_sq_per_axis[sum_sq_per_axis == 0] = 1e-12

ctr_obs = (F**2) / sum_sq_per_axis * 100.0  # procente
ctr_obs_df = pd.DataFrame(ctr_obs, index=scoruri_df.index, columns=scoruri_df.columns)
ctr_obs_df.to_csv("./dataOUT/Contributii_Observatii.csv", float_format="%.6f")
print("Contribuții observații salvat în ./dataOUT/Contributii_Observatii.csv")

# =========================================================
# (NOU) Contribuția variabilelor la axe (CTR var)
# =========================================================
L = Rxc_df.values.astype(float)  # loadings: m x p
sum_sq_load_per_axis = np.sum(L**2, axis=0, keepdims=True)
sum_sq_load_per_axis[sum_sq_load_per_axis == 0] = 1e-12

ctr_var = (L**2) / sum_sq_load_per_axis * 100.0
ctr_var_df = pd.DataFrame(ctr_var, index=Rxc_df.index, columns=Rxc_df.columns)
ctr_var_df.to_csv("./dataOUT/Contributii_Variabile.csv", float_format="%.6f")
print("Contribuții variabile salvat în ./dataOUT/Contributii_Variabile.csv")

# =========================================================
# (NOU) Plot observații în planul C1-C2
# =========================================================
if "C1" in scoruri_df.columns and "C2" in scoruri_df.columns:
    g.plot_observatii_plan(
        scoruri_df,
        c1="C1",
        c2="C2",
        title="Observații în planul componentelor principale (C1-C2)",
        save_path="./dataOUT/observatii_C1_C2.svg"
    )
    print("Plot observații C1-C2 salvat în ./dataOUT/observatii_C1_C2.svg")


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
threshold_1, j1, k1 = f.threshold(h_1)
print(f"Threshold_1: {threshold_1}, j1: {j1}, k1: {k1}")

g.dendrogram(h=h_1, labels=obs,
             title="Clasificare ierarhica observatii (method='single', metric='cityblock')",
             threshold=threshold_1, colors=None)
plt.savefig('./dataOUT/dendrogram_observatii.svg', format='svg', bbox_inches='tight')
print("Dendrogramă observații salvată!")

# Clusterizare ierarhica variabile
h_2 = hclust.linkage(y=X_std.T, method='complete', metric='correlation')
print(h_2)
threshold_2, j2, k2 = f.threshold(h_2)
print(f"Threshold_2: {threshold_2}, j2: {j2}, k2: {k2}")

g.dendrogram(h=h_2, labels=vars,
             title="Clasificare ierarhica variabile (method='complete', metric='correlation')",
             threshold=threshold_2, colors=None)
plt.savefig('./dataOUT/dendrogram_variabile.svg', format='svg', bbox_inches='tight')
print("Dendrogramă variabile salvată!")

# (4) Partiții salvate corect (fără suprascriere threshold)
labels_single = hclust.fcluster(h_1, t=threshold_1, criterion='distance').astype(int)
pd.DataFrame({"Cluster_SINGLE": labels_single}, index=obs).to_csv("./dataOUT/Clusters_SINGLE.csv")
print("Clustere SINGLE salvate în ./dataOUT/Clusters_SINGLE.csv")

labels_complete_vars = hclust.fcluster(h_2, t=threshold_2, criterion='distance').astype(int)
pd.DataFrame({"Cluster_COMPLETE_VARS": labels_complete_vars}, index=vars).to_csv("./dataOUT/Clusters_COMPLETE_VARS.csv")
print("Clustere COMPLETE_VARS salvate în ./dataOUT/Clusters_COMPLETE_VARS.csv")

# =========================================================
# (NOU) Clusterizare ierarhică OBSERVAȚII - metoda WARD
# =========================================================
# Ward folosește implicit distanța euclidiană (corect metodologic)
h_ward = hclust.linkage(y=X_std, method='ward')
print(h_ward)

# pragul pentru partiția de stabilitate maximă
threshold_w, j_w, k_w = f.threshold(h_ward)
print(f"[WARD] Threshold: {threshold_w}, j: {j_w}, k: {k_w}")

# dendrograma WARD
g.dendrogram(
    h=h_ward,
    labels=obs,
    title="Clasificare ierarhica observatii (method='ward', metric='euclidean')",
    threshold=threshold_w,
    colors=None
)
plt.savefig('./dataOUT/dendrogram_observatii_ward.svg',
            format='svg', bbox_inches='tight')
print("Dendrogramă observații WARD salvată!")

# =========================================================
# (NOU) Elbow - distanțele de agregare (WARD)
# =========================================================
g.elbow_from_linkage(
    h_ward,
    title="Elbow - distante de agregare (WARD)",
    save_path="./dataOUT/elbow_ward.svg"
)
print("Elbow (WARD) salvat!")

# =========================================================
# (NOU) Partiția finală + Silhouette
# =========================================================
# extragem etichetele de cluster folosind pragul calculat
labels_ward = hclust.fcluster(
    h_ward,
    t=threshold_w,
    criterion='distance'
).astype(int)

# salvare clustere în CSV
clusters_ward_df = pd.DataFrame(
    {"Cluster_WARD": labels_ward},
    index=obs
)
clusters_ward_df.to_csv("./dataOUT/Clusters_WARD.csv")
print("Clustere WARD salvate în ./dataOUT/Clusters_WARD.csv")

# Silhouette plot + scor mediu
sil_score = g.silhouette_plot(
    X_std,
    labels_ward,
    title="Silhouette plot (WARD)",
    save_path="./dataOUT/silhouette_ward.svg"
)
print(f"Silhouette score (WARD): {sil_score:.3f}")

g.afisare()


print("\n" + "="*50)
print("Toate graficele au fost salvate în ./dataOUT/ ca fișiere SVG!")
print("="*50)
