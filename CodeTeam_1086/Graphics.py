import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd


# portal pentru grafice in Python
# https://python-graph-gallery.com/


def corelograma(R2=None, dec=2, titlu='Corelograma',
                valMin=-1, valMax=1):
    plt.figure(num=titlu, figsize=(18 * 2, 8))
    plt.title(label=titlu, fontsize=12,
              verticalalignment='bottom', color='Blue')
    # asumam ca primit ca parametru
    # un numpy.ndarray sau pandas.DataFrame
    sb.heatmap(data=np.round(a=R2, decimals=dec),
               vmin=valMin, vmax=valMax,
               cmap='bwr', annot=True)



def intesitate_legaturi(R2=None, dec=2, titlu='Intensitate legaturi',
                        color='Oranges'):
    R2 = R2.T
    plt.figure(num=titlu, figsize=(20, 10))
    plt.title(label=titlu, fontsize=12,
              verticalalignment='bottom', color='Blue')
    # asumam ca primit ca parametru
    # un numpy.ndarray sau pandas.DataFrame
    sb.heatmap(data=np.round(a=R2, decimals=dec),
               cmap=color, annot=True, annot_kws={'size': 4, "rotation": 90})
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=8)


def cercul_corelatiilor(R2=None, V1=0, V2=1, dec=2,
                        titlu='Cercul corelatiilor'):
    # putem primi ca tip d eparametru in R2 fie un numpy.ndarray,
    # fie un pandas.DataFrame
    plt.figure(num=titlu, figsize=(8, 7))
    plt.title(label=titlu +' intre ' + 'Componenta ' + str(V1+1) + ' si ' +
              'Componenta ' + str(V2+1), fontsize=12,
              verticalalignment='bottom', color='Green')
    # generez puncte pe un cerc
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
            # plt.text(x=R2[i, V1], y=R2[i, V2], s='text')
            plt.text(x=R2[i, V1], y=R2[i, V2], color='Black',
                s='(' + str(np.round(R2[i, V1], decimals=dec)) + ', ' +
                  str(np.round(R2[i, V2], decimals=dec)) + ')')
    elif isinstance(R2, pd.DataFrame):
        plt.xlabel(xlabel=R2.columns[V1], fontsize=10,
                  verticalalignment='top', color='Blue')
        plt.ylabel(ylabel=R2.columns[V2], fontsize=10,
                   verticalalignment='top', color='Blue')
        # plt.scatter(x=R2.values[:, V1], y=R2.values[:, V2], color='Blue')
        # plt.scatter(x=R2.iloc[:].iloc[V1], y=R2.iloc[:].iloc[V2], color='Blue') # nu fuctioneaza cu slicing pe liste !!!
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
    # plt.plot(componente, valori, 'b^-')
    plt.axhline(y=1, color='Red')


def afisare():
    plt.show()