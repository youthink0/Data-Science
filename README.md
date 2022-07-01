# Data-Science

## HW1 : frequent patterns algorithm
### implement by FP-growth
![](https://github.com/youthink0/Data-Science/blob/master/HW1/FP-growth.png)
### Run
* python3 110065503_hw1.py [min support] [InputFileName] [OutputFileName] 
  * python3 110065503_hw1.py 0.2 dataset/sample.txt outputfile/output.txt

## HW2 : Kaggle comepition
### learning ML's process, ex: normalize, resampling, etc.

## HW3 : Global Optimization
### learning Global Optimization's method, including CMAES & CoDE(Differential Evolution)
### Setup
* pip install sourcedefender
### Run
* python 110065503_hw3.py

## HW4 : Graph Adversarial Attack
### implement by Nettack model
### Reference code : https://github.com/danielzuegner/nettack
### attack process
![image](https://user-images.githubusercontent.com/62932654/172207644-0a5d0f0a-735f-4c52-9523-d45d0b76653e.png)
### Setup
* create a Conda environment
  * conda create -n env_pytorch python=3.6
  * conda activate env_pytorch
* package
  * pip -r requirements.txt
### Run
* python main.py --input_file target_nodes_list.txt --data_path ./data/data.pkl --model_path saved-models/gcn.pt --use_gpu

## HW5 : Self-supervised Learning 
### implement by MERIT
### source : https://github.com/GRAND-Lab/MERIT
### Setup
* pytorch
* numpy
* networkx
* scipy
* scikit-learn ( sklearn )
### Run
* execute run.sh



