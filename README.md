# CIDAUT AI Fake Scene Classification 2024
Classify if an autonomous dirving scene is real or fake

## Quick Setup
1. Install dependancy `pip install -r requirements.txt`
2. Setup environment .env file as follows [Environment Template](./env.template):
```
MLFLOW_TRACKING_URI=<YOUR_MLFLOW_TRACKING_SERVICE_URL>
MLFLOW_TRACKING_USERNAME=<YOUR_MLFLOW_USERNAME>
MLFLOW_TRACKING_PASSWORD=<YOUR_MLFLOW_PASSWORD>
PYTHONPATH=src pytest
```
3. Setup DVC remote endpoints:
```bash
dvc remote add origin https://dagshub.com/tchiayan/aifake.dvc # Skip this part if you already add the remote endpoint

# only run below to setup the authentication locally
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <username>
dvc remote modify origin --local password <password>
```
4. Pull dataset
```
dvc pull
```
5. Reproduce train pipeline. Pipeline configuration can be check on [DVC Configuration](./dvc.yaml)
```
dvc repro
```
