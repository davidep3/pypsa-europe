import pypsa
import matplotlib.pyplot as plt
import numpy as np
n = pypsa.Network("base_s_6_elec_1h.nc")
#plotting networks
import cartopy.crs as ccrs


#collegamento BUS, evidenzio load flow linee

loading = (n.lines_t.p0.abs().mean().sort_index() / (n.lines.s_nom*n.lines.s_max_pu).sort_index()).fillna(0.)

fig,ax = plt.subplots(
    figsize = (5,5), 
    subplot_kw = {"projection" : ccrs.PlateCarree()}
    )

n.plot(ax = ax,
       
       
       bus_colors = "red",
       branch_components = ["Line"],
       line_widths = n.lines.s_nom/3e3,
       line_colors = loading,
       line_cmap = plt.cm.viridis,     #color map
       color_geomap = True,            #put the sea
       bus_sizes = 0.1,)
       
       


ax.axis("off");

#controllo capacità installata italia
plt.figure(2,figsize = (7,9))
italian_generators = n.generators[n.generators.bus.str.startswith('IT')]   #str per accedere ai metodi di stringa
italian_generators.groupby("carrier").p_nom.sum().div(1e3).plot(kind = "bar")  #GW
plt.ylabel("Installed Capacity  2019[GW]")
plt.yticks(range(0,46))



#le rinnovabili (fotovoltaico ed eolico) e CCGT tornano abbastanza con i dati di terna






#validazione carico
#andamento mensile 

total_consumption = n.loads_t.p_set.filter(like = "IT", axis = 1).sum(axis = 1).resample('H').mean()/1e3
plt.figure(3,figsize = (15,5))
plt.plot(total_consumption ,label = "2019 consumption")
plt.ylabel("Total consumption [GW]") 
plt.xlabel("Month of the year")
plt.legend(loc = "upper right")


italy_consumption = n.loads_t.p_set.filter(like = "IT", axis =1).sum(axis =1).values.sum()/1e6  #TWh
#296 TWh su Terna



#validazione flussi con estero solo frontiera Nord


plt.figure(4,figsize = (15,5))

Lines_italy = n.lines_t.p0[["1","3","4"]].sum(axis=1).resample('D').mean().div(1e3).plot(label="IMPORT/EXPORT")  #MW
#1 con l'Austria
#3 con la svizzera
#4 con la Francia


plt.legend()
plt.ylabel("GW")

import_lines = n.lines_t.p0[["1","3","4"]].sum()/1e6  #controllo sugli andamenti dei flussi di import/export per l'italia [TWh]
importazione_tot = import_lines.loc[["1", "3", "4"]].sum() 
#import italiano tot 2019 sulla frontiera Nord , in TWh , su Terna è di 38 TWh, qui 29,45 TWh, ma non ci sono tutte le interconnessioni


reference_data_Terna1 = [1.2, 21.2, 14.3]  #TWh ,da Terna; Austria,Svizzera, Francia, ho messo le foto prese da DDS2022






x = np.arange(len(import_lines)) 

# Larghezza delle barre
bar_width = 0.35

# Crea il grafico a barre
fig, ax = plt.subplots()

bars1 = ax.bar(x - bar_width / 2, import_lines.values.flatten(), width=bar_width, label='pypsa-europe 2019')
bars2 = ax.bar(x + bar_width / 2, reference_data_Terna1, width=bar_width, label='Terna 2019')

# Aggiungi le etichette e il titolo

ax.set_ylabel('Import 2019 Terna[TWh]')
ax.set_title("Comparison between pypsa-eur data and Terna data")


import_lines = import_lines.rename(index ={"1":"AT", "3":"CH", "4":"FR"})

ax.set_xticks(x)
ax.set_xticklabels(import_lines.index, rotation=45)  # Le etichette sono i nomi delle fonti
ax.legend()

# Aggiungere i valori sopra le barre
for bar in bars1:
    yval = bar.get_height()  # Ottieni l'altezza della barra
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom',fontsize =7)

for bar in bars2:
    yval = bar.get_height()  # Ottieni l'altezza della barra
    ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, round(yval, 2), ha='center', va='bottom',fontsize =7)

# Mostra il grafico
plt.tight_layout()
plt.show()


