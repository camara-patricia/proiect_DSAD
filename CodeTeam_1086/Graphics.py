import seaborn as sb
import matplotlib.pyplot as plt
import matplotlib.colors as color
import numpy as np
import scipy.cluster.hierarchy as hclust
import statsmodels.graphics.mosaicplot as smosaic
import pandas as pd


def corelograma(R2=None, dec=2, titlu='Corelograma',
                valMin=-1, valMax=1):
    # Heatmap pentru corelații/loadings cu valori [-1, 1]
    plt.figure(num=titlu, figsize=(18 * 2, 8))
    plt.title(label=titlu, fontsize=12,
              verticalalignment='bottom', color='Blue')
    sb.heatmap(data=np.round(a=R2, decimals=dec),
               vmin=valMin, vmax=valMax,
               cmap='bwr', annot=True)



def intesitate_legaturi(R2=None, dec=2, titlu='Intensitate legaturi',
                        color='Oranges'):
    # Heatmap transpus pentru vizualizarea intensității legăturilor
    R2 = R2.T
    plt.figure(num=titlu, figsize=(20, 10))
    plt.title(label=titlu, fontsize=12,
              verticalalignment='bottom', color='Blue')
    sb.heatmap(data=np.round(a=R2, decimals=dec),
               cmap=color, annot=True, annot_kws={'size': 4, "rotation": 90})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=8)


def cercul_corelatiilor(R2=None, V1=0, V2=1, dec=2,
                        titlu='Cercul corelatiilor'):
    # Cercul corelațiilor pentru 2 componente principale.
    # Variabile apropiate = corelate pozitiv, opuse = corelate negativ.
    plt.figure(num=titlu, figsize=(8, 7))
    plt.title(label=titlu +' intre ' + 'Componenta ' + str(V1+1) + ' si ' +
              'Componenta ' + str(V2+1), fontsize=12,
              verticalalignment='bottom', color='Green')
    theta = [t for t in np.arange(start=0, stop=2*np.pi, step=0.01)]
    x = [np.cos(t) for t in theta]
    y = [np.sin(t) for t in theta]
    plt.plot(x, y)
    plt.axhline(y=0, color='Green')
    plt.axvline(x=0, color='Green')

    if isinstance(R2, np.ndarray):
        plt.xlabel(xlabel='Variabila ' + str(V1+1), fontsize=10,
                  verticalalignment='top', color='Blue')
        plt.ylabel(ylabel='Variabila ' + str(V2 + 1), fontsize=10,
                   verticalalignment='top', color='Blue')
        plt.scatter(x=R2[:, V1], y=R2[:, V2], color='Red')
        for i in range(R2.shape[0]):
            plt.text(x=R2[i, V1], y=R2[i, V2], color='Black',
                s='(' + str(np.round(R2[i, V1], decimals=dec)) + ', ' +
                  str(np.round(R2[i, V2], decimals=dec)) + ')')
    elif isinstance(R2, pd.DataFrame):
        plt.xlabel(xlabel=R2.columns[V1], fontsize=10,
                  verticalalignment='top', color='Blue')
        plt.ylabel(ylabel=R2.columns[V2], fontsize=10,
                   verticalalignment='top', color='Blue')
        plt.scatter(x=R2.iloc[:, V1], y=R2.iloc[:, V2], color='Blue')
        for i in range(R2.index.size):
            plt.text(x=R2.iloc[i].iloc[V1], y=R2.iloc[i].iloc[V2], color='Black',
                        s=R2.index[i])
    else:
        raise Exception('R2 must be a pandas.DataFrame or numpy.ndarray')


# graficul valorilor proprii
def valori_proprii(valori,
        titlu='Valori proprii - varianta explicata de componentele principale'):
    plt.figure(num=titlu, figsize=(10, 6))
    plt.title(label=titlu, fontsize=12,
              verticalalignment='bottom', color='Blue')
    plt.xlabel(xlabel='Componente pricipale', fontsize=10,
               verticalalignment='top', color='Blue')
    plt.ylabel(ylabel='Valori proprii - varianta explicata', fontsize=10,
               verticalalignment='bottom', color='Blue')
    componente = ['C'+str(i+1) for i in range(valori.shape[0])]
    plt.plot(componente, valori, 'bo-')
    plt.axhline(y=1, color='Red')

# Paleta de culori pentru clustere
_COLORS = ['y', 'r', 'b', 'g', 'c', 'm', 'sienna', 'coral',
           'darkblue', 'lime', 'grey',
           'tomato', 'indigo', 'teal', 'orange', 'darkgreen']


def plot_clusters(x, y, g, groups, labels=None, title="Plot clusters"):
    # Scatter plot cu observații colorate pe clustere
    g_ = np.array(g)
    f = plt.figure(figsize=(12, 7))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=14, color='k')
    noOfGroups = len(_COLORS)
    for v in groups:
        x_ = x[g_ == v]
        y_ = y[g_ == v]
        k = int(v[1:])
        if len(x_) == 1:  # Cluster singleton
            ax.scatter(x_, y_, color='k', label=v)
        else:
            ax.scatter(x_, y_, color=_COLORS[k % noOfGroups], label=v)
    ax.legend()
    if labels is not None:
        for i in range(len(labels)):
            ax.text(x[i], y[i], labels[i])


def histograms(x, g, var):
    # Histograme separate pe clustere pentru o variabilă
    groups = set(g)
    g_ = np.array(g)
    m = len(groups)
    l = np.trunc(np.sqrt(m))
    if l * l != m:
        l += 1
    c = m // l
    if c * l != m:
        c += 1
    axes = []
    f = plt.figure(figsize=(12, 7))
    for i in range(1, m + 1):
        ax = f.add_subplot(int(l), int(c), int(i))
        axes.append(ax)
        ax.set_xlabel(var, fontsize=12, color='k')
    for v, ax in zip(groups, axes):
        y = x[g_ == v]
        ax.hist(y, bins=10, label=v, rwidth=0.9,
                range=(min(x), max(x)))
        ax.legend()


def dendrogram(h, labels=None, title='Hierarchical classification',
               threshold=None, colors=None):
    # Dendrogramă pentru clusterizare ierarhică.
    # threshold = linia de tăiere pentru partiția optimă.
    f = plt.figure(figsize=(12, 7))
    ax = f.add_subplot(1, 1, 1)
    ax.set_title(title, fontsize=14, color='k')
    if colors is None:
        hclust.dendrogram(h, labels=labels, leaf_rotation=30,
                          ax=ax, color_threshold=threshold)
    else:
        hclust.dendrogram(h, labels=labels, leaf_rotation=30, ax=ax,
                          link_color_func=lambda k: colors[k])
    if threshold is not None:
        plt.axhline(y=threshold, color='r')


def afisare():
    plt.show()

def elbow_from_linkage(h, title="Elbow (distanțe de agregare)", save_path=None):
    d = h[:, 2]
    x = np.arange(1, len(d) + 1)

    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Joncțiune (pas de agregare)")
    plt.ylabel("Distanța de agregare")
    plt.plot(x, d, marker='o')

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')


def silhouette_plot(X, labels, title="Silhouette plot", save_path=None):

    from sklearn.metrics import silhouette_samples, silhouette_score

    labels = np.array(labels)
    # Dacă ai etichete începând de la 1, e ok
    n_clusters = len(np.unique(labels))

    if n_clusters < 2:
        raise ValueError("Silhouette are sens doar pentru cel puțin 2 clustere.")

    s_avg = silhouette_score(X, labels) # scor mediu global
    s_vals = silhouette_samples(X, labels) # scoruri individuale

    plt.figure(figsize=(10, 6))
    plt.title(f"{title} | silhouette score = {s_avg:.3f}")
    plt.xlabel("Coeficient Silhouette")
    plt.ylabel("Cluster")

    y_lower = 10
    for c in np.unique(labels):
        s_c = s_vals[labels == c]
        s_c.sort()
        size_c = s_c.shape[0]
        y_upper = y_lower + size_c

        plt.fill_betweenx(np.arange(y_lower, y_upper), 0, s_c, alpha=0.7)
        plt.text(-0.05, y_lower + 0.5 * size_c, str(c))
        y_lower = y_upper + 10

    plt.axvline(x=s_avg, linestyle="--") # linie pentru scorul mediu
    plt.yticks([])
    plt.xlim([-0.2, 1.0])

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')

    return s_avg

def plot_observatii_plan(scores_df, c1='C1', c2='C2',
                         title='Observații în planul componentelor',
                         save_path=None):

    # scores_df: pandas.DataFrame cu scoruri (rânduri=observații, coloane=C1,C2,...)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 7))
    plt.title(title)
    plt.axhline(0)
    plt.axvline(0)
    plt.xlabel(c1)
    plt.ylabel(c2)

    x = scores_df[c1].values
    y = scores_df[c2].values
    plt.scatter(x, y)

    # etichete observații
    for idx in scores_df.index:
        plt.text(scores_df.loc[idx, c1], scores_df.loc[idx, c2], str(idx))

    if save_path:
        plt.savefig(save_path, format='svg', bbox_inches='tight')


