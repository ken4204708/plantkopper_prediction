# Plantkopper prediction tool

@(ITRI)[Plantkopper detection & prediction, Python]

**Plantkopper prediction tool** is a tool to automatically train and inference model via elastic regression model. Here we will train the model with different alpha. As long as the incremental of the alpha, the accuracy of the model will be decreased while the number of the feature we used will also be decreased. Our goal is to maintain the accuracy with as less as the features we used. Therefore, we define the below cost function:

>$$f_{cost} = a_m + 0.01*e_m$$ 
$a_m$ is the accuracy of the trained model 

$e_m$ is the number of the nonzero weights of the trained model

In this repo, we use the model as our inference model which can deliver maximum cost value.
--------------
## Getting Started
### Dependencies
- **Python** (>3.5)
- **numpy**(>=1.13.3)
- **scipy**(>=0.19.1)
- **scikit-learn**(>=0.20)
- **pandas**(>=0.24.2)

NOTE: Please follow the ordor of the listed modulers for installation to avoid the link/dependency of modules to be failed.
### Training mode
In this tool, you use the terminal/powershell and use the below command to perform the model training. We will automatically generate 7 different model which can predict the amount of the plantkopper from t-1 to t-7.
 ```bash
python Main.py -mode 'training' -station_name [name of climate station] -data_filename [File name of the collected data]
```
For example:
 ```bash
python Main.py -mode 'training' -station_name '芬園' -data_filename 'database.csv'
```

### Inference mode
Similar to training mode, you use the terminal/powershell and use the below command to perform the model inference. 
 ```bash
python Main.py -mode 'inference' -station_name [name of climate station] -data_filename [File name of the collected data] -model_filename [file name of the inferenced model]
```
For example:
 ```bash
python Main.py -mode 'inference' -station_name '芬園' -data_filename 'test_samples.csv' -model_filename 'ElasticNet_1.pickle'
```