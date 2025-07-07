# Udemy_MLFlow_Experiments_Wine_Data

export MLFLOW_TRACKING_URI=https://dagshub.com/<your-username>/<your-repo>.mlflow
export MLFLOW_TRACKING_USERNAME=<your-username>
export MLFLOW_TRACKING_PASSWORD=<your-personal-access-token>
python script.py


(run these codes individually in Git Bash terminal of vscode)
export MLFLOW_TRACKING_URI=https://dagshub.com/anweshabose/Udemy_MLFlow_Experiments_Wine_Data.mlflow
export MLFLOW_TRACKING_USERNAME=anweshabose
export MLFLOW_TRACKING_PASSWORD=3fa3bd226614d227631f9bb1330493a48d3f266a
python app.py

[OR]

(paste it in app.py and then run app.py directly in cmd terminal of vscode)
import dagshub
dagshub.init(repo_owner='anweshabose', repo_name='Udemy_MLFlow_Experiments_Wine_Data', mlflow=True)