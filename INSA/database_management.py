from pathlib2 import Path
import pyreadr

__path2mEsp__ = Path("databases/mEspListe.rds")

def get_mEsp_liste(path2mEsp=__path2mEsp__):
    """
    Load the species list from an RDS file and return it as a Python list.

    Assumes that the RDS file contains a single dataframe with one unnamed column.
    
    :param path2mEsp: Description
    """
    result = pyreadr.read_r(path2mEsp)
    df_mEsp = result[None]  # Extract the dataframe from the result dictionary
    liste_especes = df_mEsp[None].tolist()
    return liste_especes

if __name__ == "__main__":
    print("Testing path2mEsp: ", __path2mEsp__.resolve())
    print("Path exists: ", __path2mEsp__.exists())
    print("----------------------------")
    print("Testing get_mEsp_liste function:")

    liste_especes = get_mEsp_liste()
    print(liste_especes)
    print("Number of species loaded: ", len(liste_especes))