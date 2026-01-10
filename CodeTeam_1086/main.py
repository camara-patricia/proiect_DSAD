import pandas as pd
import Functions as f
import acp.ACP as acp
import Graphics as g

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
# le apropie le facem media 0 si ab st 1
# OBLIGATORIU IN ACP
X = tabel[vars].values
# print(X)

modelACP = acp.ACP(X)
X_std = modelACP.getXstd()
print(X_std, type(X_std), X_std.shape)
# salvati X_std in CSV file !!!!
# TODO

# extragere valori proprii - varianta explicata de componentele principale
alpha = modelACP.getAlpha()
g.valori_proprii(valori=alpha)

# extragem din model componentele principale
comp = modelACP.getComponente()
# salvati componentele principale in fisier CSV
comp_df = pd.DataFrame(data=comp,
            index=(o for o in obs),
            columns=('C'+str(j+1) for j in range(comp.shape[1])))
comp_df.to_csv('./dataOUT/ComponentePrincipale.csv')
# grafic componente principale
g.intesitate_legaturi(R2=comp_df, titlu='Conponente principale')

Rxc = modelACP.getFactorLoadings()
# salvare factor loadings in fisier CSV
# TODO
Rxc_df = pd.DataFrame(data=Rxc,
        index=(v for v in vars),
        columns=('C'+str(j+1) for j in range(comp.shape[1])))
# creare grafic corelograma factorilor de corelatie
g.corelograma(R2=Rxc_df)
# g.afisare()

# extragere scoruri (componente principale standardizate)
scoruri = modelACP.getScoruri()
# salvati scorurile in fisier CSV
# TODO
scoruri_df = pd.DataFrame(data=scoruri,
            index=(o for o in obs),
            columns=('C'+str(j+1) for j in range(comp.shape[1])))
# creare grafic scoruri
g.intesitate_legaturi(R2=scoruri_df,
        titlu='Scoruri - conponente principale standardizate',
        color='Blues')
# g.afisare()

# extragere comunalitati
comun = modelACP.getComunalitati()
# salvati comunalitatile in fisier CSV
# TODO
comun_df = pd.DataFrame(data=comun,
        index=(v for v in vars),
        columns=('C'+str(j+1) for j in range(comp.shape[1])))
# grafic al comunalitatilor
g.corelograma(R2=comun_df, titlu='Graficul comunalitatilor)')
# g.afisare()

# creati cercul corelatiior pentru primele 2 compinenete principale
g.cercul_corelatiilor(R2=Rxc_df)
g.afisare()


