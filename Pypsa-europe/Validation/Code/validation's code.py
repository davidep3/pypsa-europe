import pypsa 
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import pandas as pd
import cartopy.crs as ccrs

n= pypsa.Network("italy_2019.nc")

#collegamento BUS

def define_loading(n, pu):
    if pu:
        loading_lines = (n.lines_t.p0.abs().mean().sort_index() / (n.lines.s_nom * n.lines.s_max_pu).sort_index()).fillna(0.)
        loading_links = (n.links_t.p0.abs().mean().sort_index() / (n.links.p_nom * n.links.p_max_pu).sort_index()).fillna(0.)
    else:
        loading_lines = (n.lines_t.p0.abs().max(axis=0) / n.lines.s_nom_opt).fillna(0.)
        loading_links = (n.links_t.p0.abs().max(axis=0) / n.links.p_nom_opt).fillna(0.)
    
    return loading_lines, loading_links
    
pu = True
loading_lines, loading_links = define_loading(n, pu)

fig,ax = plt.subplots(
    figsize = (8,8), 
    subplot_kw = {"projection" : ccrs.PlateCarree()}
    )

n.plot(ax = ax,
       bus_colors = "red",
       branch_components = ["Line", "Link"],   
       line_widths = n.lines.s_nom/2e3,
       link_widths = n.links.p_nom / 2e3,
       line_colors = loading_lines,
       link_colors = loading_links,
       line_cmap = plt.cm.viridis,     #color map
       color_geomap = True,            #put the sea
       bus_sizes = 0.05,
       )
       
ax.axis("off")

# Crea la legenda personalizzata per il carico sulle linee
cmap = plt.cm.viridis

min_value = loading_lines.min()
max_value = loading_lines.max()

norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)

# Crea una lista di "patch" che corrispondono ai valori di carico
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Impostiamo un array vuoto per la colorbar

# Definisci il numero di valori da mostrare nella legenda
n_labels = 8  # Numero di etichette nella legenda

# Crea un intervallo per i valori da visualizzare
values = [min_value + i * (max_value - min_value) / (n_labels - 1) for i in range(n_labels)]

# Aggiungi la legenda
labels = [f"{v:.2f}" for v in values]  # Formatta i valori come stringhe
handles = [mpl.patches.Patch(color=cmap(norm(v))) for v in values]

ax.legend(handles, labels, title="lines load [p.u]", loc="upper right")

plt.show()


#1) validazione capacità installata

generators = n.generators[n.generators.index.str.startswith('IT')]
storage_units = n.storage_units[n.storage_units.index.str.startswith('IT')]

installed_capacity = generators.groupby("carrier").p_nom.sum()/1e3  #GW

hydro_storage = storage_units.p_nom.sum()/1e3   #GW

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

bars1 = ax.bar(x - bar_width / 2, validation_capacity.values.flatten(), width=bar_width, label='pypsa-europe 2019')
bars2 = ax.bar(x + bar_width / 2, reference_data_Terna, width=bar_width, label='Terna 2019')

# Aggiungi le etichette e il titolo
ax.set_xlabel('carrier')
ax.set_ylabel('Capacity [GW]')
ax.set_title('Comparison between pypsa-eur data and Terna data')
ax.set_xticks(x)
ax.set_xticklabels(validation_capacity.index, rotation=45)  # Le etichette sono i nomi delle fonti
ax.legend()

# Aggiungere i valori sopra le barre
for bar in bars1:
    yval = bar.get_height()  # Ottieni l'altezza della barra
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom',fontsize =8)

for bar in bars2:
    yval = bar.get_height()  # Ottieni l'altezza della barra
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom',fontsize =8)


plt.tight_layout()
plt.show()





#2) validazione carico
plt.figure(3,figsize=(15,4))

load_italy = n.loads[n.loads.index.str.startswith('IT')]
loads_t_italy = n.loads_t.p_set.loc[:, n.loads_t.p_set.columns.str.startswith('IT')]

load = loads_t_italy.sum(axis = 1).values.sum()/1e6   #294 TWh
loads_t_italy.resample("H").mean().sum(axis=1).div(1e3).plot()#andamento non disponibile su Terna
plt.ylabel("Electric demand 2019 [GW]")





#3) validazione produzione
generators_italy = n.generators[n.generators.index.str.startswith('IT')]
generators_t_italy = n.generators_t.p.loc[:, n.generators_t.p.columns.str.startswith('IT')]
storage_units_italy = n.storage_units[n.storage_units.index.str.startswith('IT')]
storage_units_t_italy = n.storage_units_t.p.loc[:, n.storage_units_t.p.columns.str.startswith('IT')]

generators= generators_t_italy.sum(axis = 1).values.sum()/1e6    #280 TWh , manca idroelettrico forse

generazione_totale = pd.concat([generators_t_italy , storage_units_t_italy], axis= 1).sum().values.sum()/1e6   #294 TWh su Terna 294 TWh alla produzione di energia totale


plt.figure(figsize=(15,3))

#termoelettrico (CCGT + coal)
CCGT = generators_t_italy.filter(like = "CCGT").sum(axis=1).resample("H").mean().div(1e3)  #GW
produzione_CCGT = CCGT.sum()/1e3  
#su DDS2022, nella foto che riporta la generazione, la produzione gas naturale è di 138 TWh, qui di 139,2 TWh

coal = generators_t_italy.filter(like="coal").sum(axis = 1).resample("H").mean().div(1e3)  #GW

termoelettrico = CCGT + coal
plt.plot(termoelettrico,label ="thermoelectric")

PV = generators_t_italy.filter(like = "solar").sum(axis=1).resample("H").mean().div(1e3) #GW
plt.plot(PV,label="Photovoltaic",color ="red",alpha = 0.1)

wind = generators_t_italy.filter(like = "wind").sum(axis = 1).resample("H").mean().div(1e3) #GW
plt.plot(wind,label ="pv",color = "green", alpha = 1)




total_generation = termoelettrico + PV + wind

# Creazione del grafico
plt.figure(figsize=(12, 6))

# Area sottostante per la produzione di CCGT
plt.fill_between(termoelettrico.index, 0, termoelettrico, color="red", label="thermoelectric", zorder=1)

# Area sottostante per la produzione fotovoltaica (PV)
plt.fill_between(PV.index, 0, PV, color="yellow", label="photovoltaic", zorder=2)

# Area sottostante per la produzione eolica (Wind)
plt.fill_between(wind.index, 0, wind, color="green", label="wind onshore", zorder=3)

# Produzione totale (somma di CCGT, PV, Wind) con l'area colorata
plt.fill_between(total_generation.index, 0, total_generation, color="blue", alpha=0, zorder=0)

# Aggiungi il grafico della produzione totale come linea per maggiore chiarezza
plt.plot(total_generation, label="Total production", color="blue", linewidth=2)

# Etichette, legenda e titolo

plt.ylabel("[GW]")

plt.legend(loc="upper left")

# Mostra il grafico
plt.tight_layout()
plt.show()


plt.show()



#grafico  a barre, per confronto con dati di Terna sulla produzione
reference_data_Terna1 = [195.7, 23.7, 20.2, 48.15]  #TWh ,da Terna; termoelettrico, fotovoltaico, eolico, idrico
termoelettrico_g = termoelettrico.sum()/1e3  # TWh
PV_g = PV.sum()/1e3  #TWh
wind_g = wind.sum()/1e3  #TWh

hydro_g = generators_t_italy.filter(like = "ror").sum(axis=1).sum()/1e6  #TWh
hydro_s = storage_units_t_italy.sum(axis=1).sum()/1e6   #TWh
hydro = hydro_g + hydro_s



generation = pd.DataFrame([termoelettrico_g, PV_g, wind_g, hydro], index=["thermoelectric", "Photovoltaic", "onshore wind", "hydro"])




x = np.arange(len(generation)) 

# Larghezza delle barre
bar_width = 0.35

# Crea il grafico a barre
fig, ax = plt.subplots()

bars1 = ax.bar(x - bar_width / 2, generation.values.flatten(), width=bar_width, label='pypsa-europe 2019')
bars2 = ax.bar(x + bar_width / 2, reference_data_Terna1, width=bar_width, label='Terna 2019')

# Aggiungi le etichette e il titolo

ax.set_ylabel('National Production[TWh]')
ax.set_title("Comparison between pypsa-eur data and Terna data")


ax.set_xticks(x)
ax.set_xticklabels(generation.index, rotation=45)  # Le etichette sono i nomi delle fonti
ax.legend()

# Aggiungere i valori sopra le barre
for bar in bars1:
    yval = bar.get_height()  # Ottieni l'altezza della barra
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom',fontsize =9)

for bar in bars2:
    yval = bar.get_height()  # Ottieni l'altezza della barra
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom',fontsize =9)

# Mostra il grafico
plt.tight_layout()
plt.show()

