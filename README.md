# INSA_2025

Author of document: N. Faure, 28 sep 2025

Project with INSA students september 2025-january 2026

Students: 
- Laetitia Guerout
- Anthony Journy

Tutor@INSA:
- Hubert Charles

Supervisors@Gentiana:
- Nicolas Faure
- François Munoz
- Alain Poirel

##  Where to start ?

- Read the reports on pCloud (.html and .pdf files at the root of "Nicolas" folder)
- Look at the output subfolder, in particuler the "Carte_donnees_mal_classees..." files, which are a prefiguration of what the project output could be.
- Look at the data, in particular the "reformated" version of InFloris database (in pCloud, subfolder databases/InFloris)

- Clone this git project
- create your own branch before committing anything.
- Install python 3.12 (3.10 would make it too, and is more convenient for other setups like pl@ntBERT)
- set up your virtual environment using the requirements.txt file

- if you just want to run the app, go to the next chapter 

- if you want to dig into the code, start looking at the scripts: 
    * main statistical modeling script: ./nf_src/statistical_modeling_main.py
    * its parameters file (an application of it may be found in pcloud, experiments/ folder): input\parameters_template.yml
    * analysis scripts in ./nf_src/model_analysis/
    * The original scripts from Benjamin Deneu: ./example*.py, and lib/* (where some scripts are from N. Faure in fact)
- Look a the results of top2000 experiment (you can download them on pCloud, it is a .csv file).
- Make your first script to visualize them. You may pick some functions from nf_src\model_analysis\prediction_analysis.py
- Later, you may load the model from "top2000" experiment and start making some inferences

CAUTION: the input files should be changed with appropriate pathes to be able to run the scripts (paths are absolute... sorry about that). In addition, I did not test the scripts since... 2024. I'll do it in the following days.

## running the app

- Make sure you have cloned the git project and got your python virtual environment set up (see above for details). If you do not want to intall the full environement and just want to run the app, a "light" version `requirements_just4app.txt` should be sufficient.
- checkout to the correct branch``git checkout Anthony_ProjectExploration`` and pull the latest version `git pull`
- Go to the root folder <INSA_2025>
- check parameter file `.\INSA\params.yml` DATA_PATH should point towards a file with species observations, prediction ranks, etc...
- run the app `streamlit run .\INSA\app_annotation.py`

Columns that should be present in the DATA_PATH file: ["PrenomNom", "Latitude", "Longitude", "rank_ground_truth", "Nom flore", "Date_Releve"]

"rank_ground_truth" can be computed using `nf_src\evaluate_rf_on_infloris.py`, but this requires a pretrained model. In the short term (Dec 2025), ask Nicolas Faure for support.





