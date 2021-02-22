#Install Docker File
docker build -t getting-started .

#Data Versioning with DVC
[DVC website](https://dvc.org)

[DVC repository](https://github.com/iterative/dvc)

## Basics

### Step 1 - Adding dvc as dependency:
poetry add DVC

### Step 2 - Start dvc:
dvc init

### Step 3 - Do commit of to initialize dvc:
git commit -m "Initialize DVC"

### Step 4 - Add data path to .gitignore:
/data/

### Step 5 - Start to tracking a data(data path + data name):
dvc add data/crm_users/training_data.csv 

### Step 6 - If you want to track the file on git use the option -f (Do this because we ignored data file):
git add -f data/crm_users/training_data.csv.dvc

### Step 7 - After make changes on data and update the tracking file:
git add data/crm_users/training_data.csv 

### Step 8 -A (Optional step, do it only if you experiment.) Retrieve an old version.
git checkout HEAD^1 data/crm_users/training_data.csv 

## Step 8 -B (Optional step, do it only if you experiment.)
dvc checkout

### Step 9 -A (Optional) Adding remote repository
dvc remote add -d storage s3://mybucket/dvcstore
###or locally
dvc remote add -d dvc-remote /tmp/dvc-storage

### Step 9 -B (Optional) 
git add .dvc/config

### Step 9 -C (Optional) 
git commit -m "Configuring remote storage."

## The best way to control data


