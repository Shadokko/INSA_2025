# joining mDataObs_V2 and result_export.csv

from pathlib2 import Path
import pandas as pd
from INSA.database_management import get_mData, get_result_export

def join_databases(path2result_export=None, path2mData=None, 
                   on=['Code_Releve', 'Nom_Valide'], 
                   columns_to_add=['rank_ground_truth', 'proba_ground_truth'],
                   name_result_export='rerun_2025'):
    if path2result_export is not None:
        result_export = get_result_export(path2result_export)
    else:
        result_export = get_result_export()
    
    if path2mData is not None:
        mData = get_mData(path2mData)
    else:
        mData = get_mData()

    result_export_subset = result_export[on + columns_to_add].drop_duplicates()
    mData_output = mData.merge(
        result_export_subset,
        how='left',
        on=on
    )

    # renaming by adding the name_result_export as suffix
    mData_output = mData_output.rename(
        columns={col: f"{col}_{name_result_export}" for col in columns_to_add}
    )

    return mData_output


if __name__ == "__main__":
    print("Testing join_databases function:")
    df_joined = join_databases()
    print("Joined DataFrame shape: ", df_joined.shape)
    print("Columns in joined DataFrame: ", df_joined.columns.tolist())


    # exporting to CSV for inspection
    output_path = Path("experiments/rerun_2025/mData_joined_result_export.csv")
    df_joined.to_csv(output_path, sep=';', index=False)
    print(f"Joined DataFrame exported to: {output_path.resolve()}")
    

    print('adding columns from another result_export, on top of the previous ones')
    df_joined_v2 = join_databases(path2result_export=Path("../cnn-sdm/experiments/top2000/Top2000_all_except_NF_default/result_export.csv"), 
                                                          path2mData=output_path, name_result_export='top2000')
    # exporting to CSV for inspection
    output_path_v2 = Path("experiments/rerun_2025/mData_joined_result_export_2023_2025.csv")
    df_joined_v2.to_csv(output_path_v2, sep=';', index=False)                                                   
    print(f"Joined DataFrame exported to: {output_path_v2.resolve()}")