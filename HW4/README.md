# Data Science HW4

## Please install anaconda & cuda 11.3 (if you want to use gpu) first.

## Create the Environment
```
conda create --name hw4 -y
conda activate hw4
conda install scipy numpy -y
```
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
```
* without gpu support
```
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

## python --version
Python 3.6.13

## package
certifi==2021.5.30
charset-normalizer==2.0.12
colorama==0.4.4
cycler==0.11.0
Cython==0.29.14
dataclasses==0.8
decorator==4.4.2
deeprobust==0.2.4
gensim==3.8.3
googledrivedownloader==0.4
idna==3.3
imageio==2.15.0
importlib-resources==5.4.0
isodate==0.6.1
Jinja2==3.0.3
joblib==1.1.0
kiwisolver==1.3.1
llvmlite==0.36.0
MarkupSafe==2.0.1
matplotlib==3.3.4
networkx==2.5.1
numba==0.53.1
numpy==1.19.5
pandas==1.1.5
Pillow==8.4.0
protobuf==3.19.4
pyparsing==3.0.9
python-dateutil==2.8.2
pytz==2022.1
PyWavelets==1.1.1
PyYAML==6.0
rdflib==5.0.0
requests==2.27.1
scikit-image==0.17.2
scikit-learn==0.24.2
scipy==1.5.4
six==1.16.0
smart-open==6.0.0
tensorboardX==2.5.1
texttable==1.6.4
threadpoolctl==3.1.0
tifffile==2020.9.3
torch==1.10.2
torch-geometric==2.0.3
torch-scatter==2.0.9
torch-sparse==0.6.12
torchvision==0.11.3
tqdm==4.64.0
typing_extensions==4.1.1
urllib3==1.26.9
wincertstore==0.2
yacs==0.1.8
zipp==3.6.0


## Run
```
python3 main.py 
```
* You can pass arguments like this.
```
python3 main.py --input_file target_nodes_list.txt --data_path ./data/data.pkl --model_path saved-models/gcn.pt
```

## Dataset
* Cora citation network
* Select the largest connected components of the graph and use 10%/10% nodes for training/validation.
* Stats:
  
| #nodes | #edges | #features | #classes |
|--------|--------|-----------|----------|
| 2485   | 10138  | 1433      | 7        |

## TODO
* attacker.py
  * implement your own attacker
* main.py
  * setup your attacker
