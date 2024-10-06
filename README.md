## Emotion Recognation Using Facial Expression Deep Learning Project

## Workflows

- Update `config.yaml`,  
- Update `secrets.yaml` [Optional],  
- Update `params.yaml`,  
- Update the entity,  
- Update the configuration manager in `src/config`,  
- Update the components,  
- Update the pipeline,  
- Update `main.py`,  
- Update `dvc.yaml`,  
- Update `app.py`,  
- How to run?


STEPS:
Clone the repository
```bash
https://github.com/rakibrohan54/EmotionRecognationClassification-Deep_Learning-Project
```
STEP 01- Create a conda environment after opening the repository
```bash
python -m venv venv
```


```bash
.\venv\Scripts\activate
```

STEP 02- install the requirements
```bash
pip install -r requirements.txt
```
MLflow

[Documentation](https://mlflow.org/docs/latest/index.html)

cmd
- mlflow ui

dagshub

MLFLOW_TRACKING_URI=https://dagshub.com/rakibrohan54/EmotionRecognationClassification-Deep_Learning-Project.mlflow
MLFLOW_TRACKING_USERNAME=rakibrohan54
MLFLOW_TRACKING_PASSWORD=68d2276306699e6c4c999c53f8494162ae6ca912
python script.py

Run as to env veriables

```bash
set MLFLOW_TRACKING_URI=https://dagshub.com/rakibrohan54/EmotionRecognationClassification-Deep_Learning-Project.mlflow
set MLFLOW_TRACKING_USERNAME=rakibrohan54
set MLFLOW_TRACKING_PASSWORD=68d2276306699e6c4c999c53f8494162ae6ca912
```


Note: Deployment Part is not included in this project.