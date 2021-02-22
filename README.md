#Install Docker File
docker build -t getting-started .

#Install kubeflow
## Step 1 - Install k8s:
sudo snap install microk8s --classic --channel=latest/edge

## Step 2 - Check your installation:
sudo microk8s status --wait-ready

## Step3 - Enable some good components(gpu is only if you have one):
microk8s enable dns dashboard gpu helm3 host-access storage istio

## Step 4 - Enable kubeflow(it take several minutes, be patient):
microk8s enable kubeflow

## Step 5 - Optional step, only if you have problem with kubeflow instalation:

microk8s.kubectl run --rm -it --restart=Never --image=ubuntu connectivity-check -- bash -c "apt update && apt install -y curl && curl https://api.jujucharms.com/charmstore/v5/~kubeflow-charmers/ambassador-88/icon.svg"

# Step 6 - Set up an easy password:
microk8s juju config dex-auth static-username=admin
microk8s juju config dex-auth static-password=yourpass


# Step 7 -Go to dashboard http://10.64.44.xip.io

