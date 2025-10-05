import pandas as pd
import numpy as np
from pathlib2 import Path

def open_flora_database(flora_database_type='bdtfx', 
                        path2database = 'databases/taxref_tracheophyta/TAXREF18.0_FR__03_05_2025.csv'):
    """
        Opens and loads a flora database file according to the specified type.

    Supported database types:
        - 'bdtfx': Reads a tab-separated UTF-8 encoded CSV file.
        - 'baseflor': Reads an Excel file (sheet 'baseflor').
        - 'taxref': Reads a semicolon-separated ISO-8859-1 encoded CSV file.
        - 'merged': Reads a semicolon-separated UTF-8 (pandas default) encoded CSV file.

    Parameters
    ----------
    flora_database_type : str, optional
        Type of flora database to open. Must be one of 'bdtfx', 'baseflor', or 'taxref'.
        Default is 'bdtfx'.
    path2database : str or Path, optional
        Path to the database file. Default is 'databases/taxref_tracheophyta/TAXREF18.0_FR__03_05_2025.csv'.

    Returns
    -------
    flora_database : pandas.DataFrame
        The loaded flora database.
    scientific_name_column : str
        The column name containing the scientific name in the database.
    valid_name_column : str
        The column name containing the valid name in the database.

    Raises
    ------
    ValueError
        If an unknown flora_database_type is provided.

    Notes
    -----
    - For 'bdtfx', the scientific and valid name column is 'Nom retenu avec auteur'.
    - For 'baseflor', the scientific and valid name column is 'NOM_SCIENTIFIQUE'.
    - For 'taxref', the scientific name column is 'NOM_COMPLET' and the valid name column is 'NOM_VALIDE'.
    - For 'merged', the scientific name column is 'NOM_COMPLET' and the valid name column is 'NOM_VALIDE'.
    """
    path2database = Path(path2database)
    # opening files
    if flora_database_type == 'bdtfx':
        # import with utf-8 encoding and tab separator
        flora_database =  pd.read_csv(path2database, sep='\t', encoding='utf-8', low_memory=False)
        scientific_name_column = 'Nom retenu avec auteur'
        valid_name_column = scientific_name_column
        nomenclatural_number_column = 'N°_Nomenclatural_BDNFF'
    elif flora_database_type == 'baseflor':
        flora_database = pd.read_excel(path2database, sheet_name='baseflor')
        scientific_name_column = 'NOM_SCIENTIFIQUE'
        valid_name_column = scientific_name_column
        nomenclatural_number_column = 'Numéro nomenclatural du nom retenu'
    elif flora_database_type == 'taxref':
        flora_database = pd.read_csv(path2database, sep=';', encoding='ISO-8859-1', low_memory=False)
        scientific_name_column = 'NOM_COMPLET'
        valid_name_column = 'NOM_VALIDE'
        nomenclatural_number_column = 'Numéro nomenclatural du nom retenu'
    elif flora_database_type == 'merged':
        flora_database = pd.read_csv(path2database, sep=';', low_memory=False)
        scientific_name_column = 'NOM_COMPLET'
        valid_name_column = 'NOM_VALIDE'
        nomenclatural_number_column = 'Numéro nomenclatural du nom retenu'
    else: # TODO: integrate taxref https://inpn.mnhn.fr/telechargement/referentielEspece/taxref/18.0/menu#
        raise ValueError(f"Unknown flora database type: {flora_database_type}. Expected 'merged', 'taxref', 'baseflor' or 'bdtfx'.")
    return flora_database, scientific_name_column, valid_name_column, nomenclatural_number_column
    


def df_filter(df,
              column='Observateur',
              n_thresh=None,
              topN=None):
    """
    filters a dataframe on the number of occurrences on a given column

    A threshold can be set on the number of instances in column (n_thresh),
    or on the top N most frequent instances (topN)

    :param df: DataFrame
    :param column: column on which to filter
    :param n_thresh: Minimal Number of instances in the column.
    :param topN: to filter on top N most frequent instances
    :return: modified dataframe
    """
    counts = np.unique(df[column].astype(str), return_counts=True)
    counts_name = pd.DataFrame({'Nom': counts[0], 'Nombre': counts[1]})

    if n_thresh is not None:
        # counting number of occurences in column_thresh
        top_instances = [name for name, counts in zip(counts_name.Nom, counts_name.Nombre) if counts >= n_thresh]
        df = df.loc[df[column].astype(str).isin(top_instances)]

    if topN is not None:
        top_instances = list(counts_name.sort_values('Nombre', ascending=False).iloc[:topN, :].Nom)
        df = df.loc[df[column].astype(str).isin(top_instances)]

    return df




