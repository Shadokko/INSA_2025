from pathlib2 import Path
import pyreadr
import re
import pandas as pd

__path2mEsp__ = Path("databases/mEspListe.rds")
__path2mData__ = Path("databases/mDataObs_V2.rds")
__path2result_export__ = Path("experiments/rerun_2025/result_export.csv")

def get_result_export(path2result_export=__path2result_export__):
    """
    Load the result_export CSV file and return it as a pandas DataFrame.
    
    :param path2result_export: Description
    """
    df_result_export = pd.read_csv(path2result_export, sep=';')
    return df_result_export

def get_mEsp_liste(path2mEsp=__path2mEsp__, remove_date=True):
    """
    Load the species list from an RDS file and return it as a Python list.

    Assumes that the RDS file contains a single dataframe with one unnamed column.
    
    :param path2mEsp: Description
    """
    result = pyreadr.read_r(path2mEsp)
    df_mEsp = result[None]  # Extract the dataframe from the result dictionary
    liste_especes = df_mEsp[None].tolist()
    liste_especes[0] = 'Aucune'  # Instead of 'Relevé'
    if remove_date:
        liste_especes = [re.split(',', espece)[0] for espece in liste_especes]
    return liste_especes

def get_mData(path2mData=__path2mData__):
    """
    Load the main data from an RDS file and return it as a pandas DataFrame.
    
    :param path2mData: Description
    """
    result = pyreadr.read_r(path2mData)
    df_mData = result[None]  # Extract the dataframe from the result dictionary
    return df_mData

if __name__ == "__main__":
    print("Testing path2mEsp: ", __path2mEsp__.resolve())
    print("Path exists: ", __path2mEsp__.exists())
    print("----------------------------")
    print("Testing get_mEsp_liste function:")

    liste_especes = get_mEsp_liste()
    print(liste_especes)
    print("Number of species loaded: ", len(liste_especes))

    print("----------------------------")
    print("Testing path2mData: ", __path2mData__.resolve())
    print("Path exists: ", __path2mData__.exists())
    print("----------------------------")
    print("Testing get_mData function:")
    df_mData = get_mData()
    print(df_mData.head())
    print("Number of rows in mData: ", len(df_mData))
    path2csv = __path2mData__.parent / (__path2mData__.stem + ".csv")
    print("Saving mData to CSV at: ", path2csv.resolve())
    df_mData.to_csv(path2csv, sep=';', index=False)
    print("Done !")

