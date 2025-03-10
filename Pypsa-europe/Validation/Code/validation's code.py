import pypsa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd



n = pypsa.Network("/home/davide/pypsa-eur/results/Italy2019_10.1/networks/base_s_10_elec_1h.nc")
#plotting networks
import cartopy.crs as ccrs

#collegamento BUS

line_loading = (n.lines_t.p0.abs().mean().sort_index() / (n.lines.s_nom*n.lines.s_max_pu).sort_index()).fillna(0.)
link_loading = (n.links_t.p0.abs().mean().sort_index() / (n.links.p_nom*n.links.p_max_pu).sort_index()).fillna(0.)

n.links_t.p0.abs().mean()

fig,ax = plt.subplots(
    figsize = (8,8), 
    subplot_kw = {"projection" : ccrs.PlateCarree()}
    )

n.plot(ax = ax,
       
       
       
       bus_colors = "red",
       branch_components = ["Line","Link"],
       line_widths = n.lines.s_nom/3e3,
       line_colors = line_loading,
       line_cmap = plt.cm.viridis,     #color map
       color_geomap = True,            #put the sea
       bus_sizes = 0.1,)
       
       


ax.axis("off")

# Crea la legenda personalizzata per il carico sulle linee
cmap = plt.cm.viridis
norm = mpl.colors.Normalize(vmin=loading.min(), vmax=loading.max())

# Crea una lista di "patch" che corrispondono ai valori di carico
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Impostiamo un array vuoto per la colorbar

# Definisci il numero di valori da mostrare nella legenda
n_labels = 5  # Numero di etichette nella legenda

# Crea un intervallo per i valori da visualizzare
values = [loading.min() + i * (loading.max() - loading.min()) / (n_labels - 1) for i in range(n_labels)]

# Aggiungi la legenda
labels = [f"{v:.2f}" for v in values]  # Formatta i valori come stringhe
handles = [mpl.patches.Patch(color=cmap(norm(v))) for v in values]

ax.legend(handles, labels, title="loading [p.u]", loc="upper right")

plt.show()






#1) validazione capacità installata

installed_capacity = n.generators.groupby("carrier").p_nom.sum()/1e3  #GW

hydro_storage = n.storage_units.p_nom.sum()/1e3   #GW

hydro_storage_s = pd.Series([hydro_storage], index=["hydro_storage"])


df = pd.concat([installed_capacity, hydro_storage_s])

df["hydropower"] = df["ror"] + df["hydro_storage"]

validation_capacity = df.loc[["CCGT", "coal", "onwind" ,"solar", "hydropower"]]  



#confronto con dati Terna CAPACITA' INSTALLATA


reference_data_Terna = [40.5, 10, 10.7, 20.8, 19.8]  # Presi da Terna, CCGT preso sia con produzione di calore che senza, carobne da DDS 2019


#grafico confronto capacità
carriers = validation_capacity.index

# Posizioni per le barre (per evitare sovrapposizioni)
x = np.arange(len(carriers))

# Dati per le barre
bar_width = 0.35  # larghezza delle barre

# Crea il grafico a barre
fig, ax = plt.subplots()

bars1 = ax.bar(x - bar_width / 2, validation_capacity.values.flatten(), width=bar_width, label='Capacità installata 2019')
bars2 = ax.bar(x + bar_width / 2, reference_data_Terna, width=bar_width, label='Dati Terna 2019')

# Aggiungi le etichette e il titolo
ax.set_xlabel('carrier')
ax.set_ylabel('Capacità [GW]')
ax.set_title('Confronto Capacità Installata pypsa-eur vs Dati di Terna')
ax.set_xticks(x)
ax.set_xticklabels(validation_capacity.index, rotation=45)  # Le etichette sono i nomi delle fonti
ax.legend()

# Aggiungere i valori sopra le barre
for bar in bars1:
    yval = bar.get_height()  # Ottieni l'altezza della barra
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()  # Ottieni l'altezza della barra
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom')


plt.tight_layout()
plt.show()












#2) validazione carico
plt.figure(3,figsize=(15,4))
load = n.loads_t.p_set.sum(axis = 1).values.sum()/1e6   #294 TWh
n.loads_t.p.resample("H").mean().sum(axis=1).div(1e3).plot()     #andamento non disponibile su Terna






#3) validazione produzione
generators= n.generators_t.p.sum(axis = 1).values.sum()/1e6    #279 TWh , manca idroelettrico forse

generazione_totale = pd.concat([n.generators_t.p , n.storage_units_t.p], axis= 1).sum().values.sum()/1e6   #294 TWh su Terna 294 TWh alla produzione di energia totale


plt.figure(figsize=(15,3))

#termoelettrico (CCGT + coal)
CCGT = n.generators_t.p.filter(like = "CCGT").sum(axis=1).resample("H").mean().div(1e3)  #GW

coal = n.generators_t.p.filter(like="coal").sum(axis = 1).resample("H").mean().div(1e3)  #GW

termoelettrico = CCGT + coal
plt.plot(termoelettrico,label ="termoelettrico")

PV = n.generators_t.p.filter(like = "solar").sum(axis=1).resample("H").mean().div(1e3) #GW
plt.plot(PV,label="PV",color ="red",alpha = 0.1)

wind = n.generators_t.p.filter(like = "wind").sum(axis = 1).resample("H").mean().div(1e3) #GW
plt.plot(wind,label ="pv",color = "green", alpha = 1)




total_generation = termoelettrico + PV + wind

# Creazione del grafico
plt.figure(figsize=(12, 6))

# Area sottostante per la produzione di CCGT
plt.fill_between(termoelettrico.index, 0, termoelettrico, color="blue", alpha=0.3, label="termoelettrico", zorder=1)

# Area sottostante per la produzione fotovoltaica (PV)
plt.fill_between(PV.index, 0, PV, color="red", alpha=0.3, label="PV", zorder=2)

# Area sottostante per la produzione eolica (Wind)
plt.fill_between(wind.index, 0, wind, color="green", alpha=0.3, label="Wind", zorder=3)

# Produzione totale (somma di CCGT, PV, Wind) con l'area colorata
plt.fill_between(total_generation.index, 0, total_generation, color="grey", alpha=0.5, label="Totale", zorder=0)

# Aggiungi il grafico della produzione totale come linea per maggiore chiarezza
plt.plot(total_generation, label="Totale", color="black", linewidth=2)

# Etichette, legenda e titolo
plt.xlabel("Data e Ora")
plt.ylabel("[GW]")

plt.legend(loc="upper left")

# Mostra il grafico
plt.tight_layout()
plt.show()



plt.show()



#grafico per confronto con dati di Terna
reference_data_Terna1 = [195.7, 23.7, 20.2]  #TWh ,da Terna
termoelettrico_g = termoelettrico.sum()/1e3  # TWh
PV_g = PV.sum()/1e3
wind_g = wind.sum()/1e3

generation = pd.DataFrame([termoelettrico_g, PV_g, wind_g], index=["termoelettrico", "PV", "wind"])


x = np.arange(len(generation)) 

# Larghezza delle barre
bar_width = 0.35

# Crea il grafico a barre
fig, ax = plt.subplots()

bars1 = ax.bar(x - bar_width / 2, generation.values.flatten(), width=bar_width, label='Produzione 2019')
bars2 = ax.bar(x + bar_width / 2, reference_data_Terna1, width=bar_width, label='Dati Terna 2019')

# Aggiungi le etichette e il titolo
ax.set_xlabel('Fonti')
ax.set_ylabel('Produzione Totale [TWh]')
ax.set_title('Confronto Produzione pypsa-eur vs Dati di Terna')
ax.set_xticks(x)
ax.set_xticklabels(generation.index, rotation=45)  # Le etichette sono i nomi delle fonti
ax.legend()

# Aggiungere i valori sopra le barre
for bar in bars1:
    yval = bar.get_height()  # Ottieni l'altezza della barra
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom')

for bar in bars2:
    yval = bar.get_height()  # Ottieni l'altezza della barra
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom')

# Mostra il grafico
plt.tight_layout()
plt.show()

