"""
Generating species-label correspondance
"""

import pandas as pd
import numpy as np
from pathlib2 import Path

path2dataset = Path(r'C:\Users\fauren\PycharmProjects\cnn-sdm\data\full_dataset.csv')

sdm = pd.read_csv(path2dataset, sep=';')

list_species = []
for label in np.unique(sdm.Label):
    list_species.append(sdm.loc[sdm.Label == label, 'Species'].values[0])

table_sp_label = pd.DataFrame({'sdm_label': np.unique(sdm.Label), 'Species': list_species})
table_sp_label.to_csv(path2dataset.parent / 'species_sdmlabel_table.csv', sep=';', index=False)

print('Done !')

