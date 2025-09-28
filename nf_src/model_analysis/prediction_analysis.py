"""
Analysing model predictions
"""

from pathlib2 import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotnine as gg
from gentiana_toolbox.database import df_filter
import sys

path2folder2analyze = Path(r'C:\Users\fauren\Documents_nosync\cnn_sdm_output\experiments\Compare_models')
path2folder2analyze_top1200 = path2folder2analyze.parent / 'top1200'

path2preds_rf_if = Path(r'C:\Users\fauren\Documents_nosync\cnn_sdm_output\preds_rf_infloris_14-test_set_40000_small.csv')
# Path(r'C:\Users\fauren\PycharmProjects\cnn-sdm\output\preds_rf_infloris_14-test_set_small.csv')

path2outputfolder = Path(r'C:\Users\fauren\Documents_nosync\bac_a_sable_GP_databases\output')


def concat_melt_predictions(path2folder2analyze, glob='*/result_export.csv', remove_top100=True):
    for index, path2preds in enumerate(path2folder2analyze.glob(glob)):
        print(f'loading: {path2preds}')
        preds = pd.read_csv(path2preds, sep=';')
        if index == 0:
            if remove_top100: concat_preds = preds.iloc[:, :-100]
            else: concat_preds = preds
        concat_preds[f'rang {path2preds.parent.name}'] = preds['rank_ground_truth']

    melt_columns = [column for column in concat_preds.columns if 'rang ' in column]

    melt_preds = pd.melt(concat_preds, value_vars=melt_columns,
                         var_name='Modèle',
                         id_vars=['Note', 'Nom_Valide', 'Observateur'],
                         value_name='Rang')

    return concat_preds, melt_preds

concat_preds, melt_preds = concat_melt_predictions(path2folder2analyze)

newnames = {'rang rf_train_if': 'random forest, sur patch 1x1, train set InFloris',
            'rang rf_train_if_3': 'random forest, sur patch 2x2, train set InFloris',
            'rang rf_train_if_4': 'random forest, sur patch 4x4, train set InFloris',
            'rang rf_train_if_8': 'random forest, sur patch 8x8, train set InFloris',
            'rang rf_train_if_4_all_but_test': 'random forest, patch 4x4, entraîné sur TOUTES les données InFloris SAUF test',
            'rang rf_train_if_4_all': 'idem (random forest 4x4), mais entraîné sur toutes données y compris test',
            'rang rf_train_if_4_all_but_test_reg': "random forest régularisé 4x4, entraîné sur TOUTES les données InFloris SAUF test",
            'rang rf_train_if_4_all_reg': "idem (random forest régularisé 4x4), mais entraîné également sur données de test",
            'rang rf_train_cnnsdm': 'random forest, sur patch 1x1, train set cnn-sdm',
           'rang freq_train_if': "rang selon la fréquence dans le train set InFloris, sans info géographique",
            'rang random': 'aléatoire: tirage parmi les 4520 espèces sans pondération'}

category_orders = {'Modèle': ['rang random', 'rang freq_train_if', 'rang rf_train_cnnsdm',
                              'rang rf_train_if', 'rang rf_train_if_3', 'rang rf_train_if_4', 'rang rf_train_if_8',
                              'rang rf_train_if_4_all_but_test', 'rang rf_train_if_4_all', 'rang rf_train_if_4_all_but_test_reg']}

###############################################
# performances par note pour le modèle régularisé
fig4 = px.box(melt_preds.loc[melt_preds['Modèle'] == 'rang rf_train_if_4_all_but_test_reg', :],
              x='Note', y='Rang', log_y=False,
             hover_data=['Nom_Valide', 'Observateur', 'Note'], template="seaborn",
              category_orders=category_orders, width=1150, height=800)
# fig4.update_layout(title={'xanchor': 'center', 'yanchor': 'top'})
fig4.update_layout(
    title="Correspondance modèle 'random forest régularisé' et note 'Alain poirel'"
          f"<br><sup>Ordonnée: rang de la vérité terrain (espèce effectivement observée) selon le classement de probabilités du modèle. "
          f"<br>Abscisse: note selon le modèle développé par Alain Poirel (-1 -> données non notées). "
          f"<br>Entrainement: toutes données d'Infloris, sauf données d'évaluation"
          f"<br>Modèles évalués sur données A. Poirel, J-M Tison et A. Mas ({len(concat_preds)} données InFloris)</sup>",
    xaxis_title="Notes 'Alain'",
    yaxis_title="Rang de la prédiction du modèle, pour l'espèce effectivement vue (vérité terrain)",
)
fig4.update_layout(xaxis = dict(tickmode='array',tickvals=list(range(-1, 10))))
fig4.update_layout(margin=dict(l=20, r=20, t=180, b=20))
fig4.write_html(str(path2outputfolder / ('Perf_model_note.html')))
fig4.show()

###################################################
# performances par modèle et par observateur
fig5 = px.box(melt_preds, x='Observateur', y='Rang', log_y=False, color='Modèle',
             hover_data=['Nom_Valide', 'Observateur', 'Note'], template="seaborn",
              category_orders=category_orders, width=1200, height=800)
fig5.update_layout(
    title="Prédictions des différents modèles selon observateur"
          f"<br><sup>Entrainement: 'train set InFloris' données F.Gourgues, M. Kopf et B. Grange (env. 230000 données); ou 'train set cnn-sdm': données thèse B. Deneu (env. 80000 données)"
          f"<br>Modèles évalués sur données A. Poirel, J-M Tison et A. Mas ({len(concat_preds)} données InFloris)</sup>",    xaxis_title="Observateur",
    yaxis_title="Rang de la prédiction du modèle, pour l'espèce effectivement vue (vérité terrain)",
)
fig5.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     ))
fig5.update_layout(xaxis = dict(tickmode='array',tickvals=list(range(-1, 10))))
fig5.update_layout(margin=dict(l=20, r=20, t=150, b=20))
fig5.write_html(str(path2outputfolder / ('Perf_models_observers_lin.html')))
fig5.show()

# ===========================
# Analyse des modèles Top1200

concat_preds, melt_preds = concat_melt_predictions(path2folder2analyze_top1200)

# performances du modèle de référence sur le jeu de test
fig1200 = px.box(melt_preds,
              x='Observateur', y='Rang', color='Observateur', log_y=False,
             hover_data=['Nom_Valide', 'Observateur', 'Note'], template="seaborn",
              category_orders=category_orders, width=1150, height=800)
# fig1200.update_layout(title={'xanchor': 'center', 'yanchor': 'top'})
fig1200.update_layout(
    title="Performance du modèle 'de référence' à 1200 taxons"
          f"<br><sup>Ordonnée: rang de la vérité terrain (espèce effectivement observée) selon le classement de probabilités du modèle. "
          f"<br>Abscisse: note selon le modèle développé par Alain Poirel (-1 -> données non notées). "
          f"<br>Entrainement: toutes données d'Infloris, sauf données d'évaluation"
          f"<br>Modèles évalués sur données A. Poirel, J-M Tison et A. Mas ({len(concat_preds)} données InFloris)</sup>",
    xaxis_title="Notes 'Alain'",
    yaxis_title="Rang de la prédiction du modèle, pour l'espèce effectivement vue (vérité terrain)",
)
fig1200.update_layout(xaxis = dict(tickmode='array',tickvals=list(range(-1, 10))))
fig1200.update_layout(margin=dict(l=20, r=20, t=180, b=20))
fig1200.write_html(str(path2outputfolder / ('Perf_model_1200.html')))
fig1200.show()

# Visualisation des données mal classées Alain Poirel
fig1 = px.scatter_mapbox(concat_preds.loc[(concat_preds.Observateur == 'Alain POIREL') &
                                          (concat_preds.rank_ground_truth > 600), :], size_max=10,
                         lat="Latitude", lon="Longitude", color='rank_ground_truth',
                         hover_name="Nom_Valide", hover_data=["Observateur", "Date_Releve", 'NbObs_Releve'],
                         zoom=8, width=1150, height=800)
fig1.update_layout(title="Données d'Alain Poirel mal classées par le modèle à 1200 espèces")
fig1.update_layout(mapbox_style="open-street-map")
fig1.write_html(str(path2outputfolder / ('Carte_donnees_mal_classees_Alain_Poirel.html')))
fig1.show()

concat_preds, melt_preds = concat_melt_predictions(path2folder2analyze_top1200,
                                                   glob='*/result_export_all.csv',
                                                   remove_top100=False)

# séparer les données test et train
concat_preds['test set'] = concat_preds.Observateur.isin(['Alain POIREL', 'Jean-Marc TISON', 'Anaïs MAS'])

# performances sur le jeu de train et test
fig1200 = px.box(df_filter(concat_preds, topN=10),
              x='Observateur', y='rank_ground_truth', color='test set', log_y=False,
             hover_data=['Nom_Valide', 'Observateur', 'Note'],
                 template="seaborn", width=950, height=600)
# fig1200.update_layout(title={'xanchor': 'center', 'yanchor': 'top'})
fig1200.update_layout(
    title="Performance du modèle 'de référence' à 1200 taxons"
          f"<br><sup>Ordonnée: rang de la vérité terrain (espèce effectivement observée) selon le classement de probabilités du modèle. "
          f"<br>Abscisse: Observateur (couleur selon train ou test set)"
          f"<br>Entrainement: toutes données d'Infloris, sauf données de test (A. Poirel, J-M Tison et A. Mas)</sup>",
    xaxis_title="Observateur",
    yaxis_title="Rang de la prédiction du modèle, pour l'espèce effectivement vue (vérité terrain)",
)
fig1200.update_layout(margin=dict(l=20, r=20, t=150, b=20))
fig1200.write_html(str(path2outputfolder / ('Perf_model_1200_all.html')))
fig1200.show()

# performances du modèle 1200 selon note
fig4 = px.box(concat_preds.loc[concat_preds['test set'] == True, :],
              x='Note', y='rank_ground_truth', log_y=False,
             hover_data=['Nom_Valide', 'Observateur', 'Note'], template="seaborn",
              category_orders=category_orders, width=1150, height=800)
# fig4.update_layout(title={'xanchor': 'center', 'yanchor': 'top'})
fig4.update_layout(
    title="Correspondance modèle 'random forest régularisé' et note 'Alain poirel'"
          f"<br><sup>Ordonnée: rang de la vérité terrain (espèce effectivement observée) selon le classement de probabilités du modèle. "
          f"<br>Abscisse: note selon le modèle développé par Alain Poirel (-1 -> données non notées). "
          f"<br>Entrainement: toutes données d'Infloris, sauf données d'évaluation"
          f"<br>Résultats sur données de test uniquement: données A. Poirel, J-M Tison et A. Mas "
          f"({len(concat_preds.loc[concat_preds['test set'] == True, :])} données InFloris)</sup>",
    xaxis_title="Note donnée par le modèle développé par Alain Poirel",
    yaxis_title="Rang de la prédiction du modèle, pour l'espèce effectivement vue (vérité terrain)",
)
fig4.update_layout(xaxis = dict(tickmode='array',tickvals=list(range(-1, 10))))
fig4.update_layout(margin=dict(l=20, r=20, t=180, b=20))
fig4.write_html(str(path2outputfolder / ('Perf_model_1200_note.html')))
fig4.show()

# Visualisation des données mal classées tous utilisateurs
concat_preds['Atypicité observation'] = ((concat_preds.rank_ground_truth - 200) / 50).astype('int')
concat_preds.loc[concat_preds['Atypicité observation'] < 0, 'Atypicité observation'] = 0
fig1 = px.scatter_mapbox(df_filter(concat_preds.loc[concat_preds.rank_ground_truth > 300, :], topN=50),
                         size='Atypicité observation',
                         lat="Latitude", lon="Longitude", color='Observateur',
                         hover_name="Nom_Valide", hover_data=["Observateur", "Date_Releve", 'NbObs_Releve'],
                         zoom=8, width=1150, height=800)
fig1.update_layout(title="Données avec taille selon 'atypicité' prédite par le modèle")
fig1.update_layout(mapbox_style="open-street-map")
fig1.write_html(str(path2outputfolder / ('Carte_donnees_mal_classees_tous_utilisateurs.html')))
fig1.show()


print('Done !')
sys.exit(0) # the rest is only archive

fig = px.box(preds, x='Note', y='rf_note',
             hover_data=['Nom_Valide', 'Observateur', 'rank_preds', 'Note'], points="all")
fig.update_layout(
    title="Comparaison des notes 'Alain' et des notes du modèle random_forest "
          f"<br>Modèle random forest entraîné sur les données F. Gourgues, et évalué sur un échantillon de {len(preds)} autres données InFloris Infloris",
    xaxis_title="Notes 'Alain'",
    yaxis_title="Notes 'random forest' pour la donnée",
)
fig.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = list(range(-1, 10))
    ),
    yaxis=dict(
        tickmode='array',
        tickvals=list(range(11))
    )
)
fig.show()

fig2 = px.box(preds, x='Note', y='rank_preds', log_y=True,
             hover_data=['Nom_Valide', 'Observateur', 'rank_preds', 'Note'],
             points="all")
fig2.update_layout(
    title="Comparaison des notes 'Alain' et des rangs de prédiction du modèle random_forest "
          f"<br>Modèle random forest entraîné sur les données F. Gourgues, et évalué sur un échantillon de {len(preds)} autres données InFloris Infloris",
    xaxis_title="Notes 'Alain'",
    yaxis_title="Rang de la prédiction du modèle 'random forest', pour l'espèce effectivement vue (vérité terrain)",
)
fig2.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = list(range(-1, 10))
    )
)
fig2.show()

preds['aleatoire'] = (4520 * np.random.random(len(preds))).astype('int64') + 1
fig3 = px.box(preds, x='Note', y='aleatoire', log_y=True,
             hover_data=['Nom_Valide', 'Observateur', 'rank_preds', 'Note'],
             points="all")
fig3.update_layout(
    title="Comparaison des notes 'Alain' et des rangs de prédiction d'un modèles ALEATOIRE"
          f"<br>Le modèle tire une prédiction aléatoire (équiprobable) parmi les 4520 possibles, pour les {len(preds)} données",
    xaxis_title="Notes 'Alain'",
    yaxis_title="Rang de la prédiction du modèle aléatoire, pour l'espèce effectivement vue (vérité terrain)",
)
fig3.update_layout(
    xaxis = dict(
        tickmode = 'array',
        tickvals = list(range(-1, 10))
    )
)
fig3.show()

# comparing models
melt_preds = pd.melt(preds, value_vars=['aleatoire', 'rank_preds_sdmdataset_model', 'rank_freqs', 'rank_preds'], var_name='Modèle',
                     id_vars=['Note', 'Nom_Valide', 'Observateur'], value_name='Rang')

fig4 = px.box(melt_preds, x='Note', y='Rang', log_y=False, color='Modèle',
             hover_data=['Nom_Valide', 'Observateur', 'Note'], template="seaborn")
# fig4.update_layout(title={'xanchor': 'center', 'yanchor': 'top'})
fig4.update_layout(
    title="Comparaison de 4 modèles: prédictions random_forest; prédiction selon la fréquence; tirage aléatoire équiprobable"
          f"<sup><br><br>Ordonnée: rang de la vérité terrain (espèce effectivement observée) selon le classement de probabilités du modèle. "
          f"Abscisse: note selon le modèle développé par Alain Poirel (-1 -> données non notées). "
          f"<br>Modèles appris sur les données de Frédéric Gourgues seulement (192 000 données), évalués sur un échantillons aléatoire de {len(preds)} autres données InFloris</sup>",
    xaxis_title="Notes 'Alain'",
    yaxis_title="Rang de la prédiction du modèle, pour l'espèce effectivement vue (vérité terrain)",
)
newnames = {'rank_preds': 'random forest (entrainées sur données InFloris, F. Gourgues)',
            'rank_preds_sdmdataset_model': 'random forest (entrainé sur données Thèse B. Deneu)',
           'rank_freqs': "fréquence (rang selon la fréquence, sans info géographique)",
            'aleatoire': 'aléatoire (tirage parmi les 4520 espèces, sans pondération)'}
fig4.for_each_trace(lambda t: t.update(name = newnames[t.name],
                                      legendgroup = newnames[t.name],
                                      hovertemplate = t.hovertemplate.replace(t.name, newnames[t.name])
                                     ))

fig4.update_layout(xaxis = dict(tickmode='array',tickvals=list(range(-1, 10))))
fig4.update_layout(margin=dict(l=20, r=20, t=150, b=20))
fig4.show()

topN = 50
fig5 = px.box(df_filter(preds, topN=topN), x='Observateur', y='rank_preds', log_y=True,
             hover_data=['Nom_Valide', 'Observateur', 'rank_preds', 'Note'])
fig5.update_layout(
    title="Prédictions du modèle random forest selon observateur"
          f"<br>Modèle random forest entraîné sur les données F. Gourgues, et évalué sur un échantillon de {len(df_filter(preds, topN=50))} autres données InFloris Infloris",
    xaxis_title="Observateur",
    yaxis_title="Rang de la prédiction du modèle 'random forest', pour l'espèce effectivement vue (vérité terrain)",
)
fig5.show()

fig6 = px.box(preds, x='Annee', y='rank_preds', log_y=True,
             hover_data=['Nom_Valide', 'Observateur', 'rank_preds', 'Note'])
fig6.update_layout(
    title="Prédictions du modèle random forest selon année"
          f"<br>Modèle random forest entraîné sur les données F. Gourgues, et évalué sur un échantillon de {len(preds)} autres données InFloris Infloris",
    xaxis_title="Année",
    yaxis_title="Rang de la prédiction du modèle 'random forest', pour l'espèce effectivement vue (vérité terrain)",
)
fig6.show()

fig7 = px.histogram(preds, x='rank_preds', opacity=0.2, nbins=4520, cumulative=True, template="seaborn")
fig7.update_layout(
    title="Histogramme cumulé des prédictions du modèle random forest"
          f"<br>Modèle random forest entraîné sur les données F. Gourgues, et évalué sur un échantillon de {len(preds)} autres données InFloris Infloris",
    xaxis_title="Rang")
fig7.show()

gg.options.figure_size = (16, 9)
plot5 = gg.ggplot(melt_preds, gg.aes(x='Rang', color='Modèle', fill='Modèle')) \
        + gg.geom_density(alpha=0.2) + gg.facet_wrap('~ Note') + gg.scale_y_sqrt()
print(plot5)


print('Done !')