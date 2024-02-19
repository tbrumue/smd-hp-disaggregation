# Disaggregation of Heat Pump Load Profiles From Low-Resolution Smart Meter Data

### Authors
- Tobias Brudermüller (Brudermueller), Bits to Energy Lab, ETH Zurich: <tbrudermuell@ethz.ch>
- Fabian Breer, Chair for Electrochemical Energy Conversion and Storage Systems, RWTH Aachen University: <fabian.breer@isea.rwth-aachen.de>

This repository contains the Python code and data for the [following paper](https://dl.acm.org/doi/abs/10.1145/3600100.3623731): 

> *Tobias Brudermueller, Fabian Breer, and Thorsten Staake. 2023. Disaggregation of Heat Pump Load Profiles From Low-Resolution Smart Meter Data. In Proceedings of the 10th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation (BuildSys '23). Association for Computing Machinery, New York, NY, USA, 228–231.*

For detailed explanations about underlying assumptions, and implementations please refer to this source. 

**If you make use of this repository or paper, please use the following citation**: 

```
@inproceedings{10.1145/3600100.3623731, 
author = {Brudermueller, Tobias and Breer, Fabian and Staake, Thorsten}, 
title = {Disaggregation of Heat Pump Load Profiles From Low-Resolution Smart Meter Data}, 
year = {2023}, 
isbn = {9798400702303}, 
publisher = {Association for Computing Machinery}, 
address = {New York, NY, USA}, 
url = {https://doi.org/10.1145/3600100.3623731}, 
doi = {10.1145/3600100.3623731}, 
booktitle = {Proceedings of the 10th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation}, 
pages = {228–231}, 
numpages = {4}, 
keywords = {Smart Meter Data, Non-Intrusive Load Monitoring, Low Resolution, Load Disaggregation, Heat Pump Optimization, Energy Efficiency}, 
location = {Istanbul, Turkey}, 
series = {BuildSys '23}
}
```

---

### Abstract 

As the number of heat pumps installed in residential buildings increases, their energy-efficient operation becomes increasingly important to reduce costs and ensure the stability of the power grid. The deployment of smart electricity meters results in large amounts of smart meter data that can be used for heat pump optimization. However, sub-metering infrastructure to monitor heat pumps’ energy consumption is costly and rarely available in practice. Non-intrusive load monitoring addresses this issue and disaggregates appliance-level consumption from aggregate measurements. However, previous studies use high-resolution data of active and reactive power and do not focus on heat pumps. In this context, our study is the first to disaggregate heat pump load profiles using commonly available smart meter data with energy measurements at 15-minute resolution. We use a sliding-window approach to train and test deep learning models on a real-world data set of 363 Swiss households with heat pumps observed over a period of 8 years. Evaluating our approach with a 5-fold cross-validation, our best model achieves a mean R2 score of 0.832 and an average RMSE of 0.169 kWh, which is similar to previous work that uses high-resolution measurements of active and reactive power. Our algorithms enable real-world applications to monitor the energy efficiency of heat pumps in operation and to estimate their flexibility for demand response programs.

---

### Installation 

If you want to use your Python interpreter directly, please check that you have the packages pip-installed which are listed in the file ```installation/requirements.yml```. Otherwise, if you want to create an anaconda environment named ```hp_diagg```, you can use the following commands.

1. Navigate to the installation folder: ```cd <path to this repo>/installation/```
2. Installing environment: ```conda env create -f requirements_mac.yml``` if you are on MacOS or ```conda env create -f requirements_otherwise.yml``` otherwise
3. Open environment for a session: ```conda activate hp_disagg```
4. [...] Run whatever code (e.g. use notebooks provided) [...]
5. Close environment after a session: ```conda deactivate```

**Please also check out the notes on the usage of ```tensorflow-metal``` on Apple silicon chips at the bottom of this ```README```-file!!! Its use may lead to problems with running the code in this repo!**

---

### Usage 

Probably, the best way to understand this repository is to proceed in the following order: 

1. Read the [paper](https://dl.acm.org/doi/abs/10.1145/3600100.3623731), which describes the methodology in detail.
2. Skim the data provided in the ```data``` folder. 
3. Check out the notebooks provided in the ```notebooks``` folder. 
4. Apply the methods to your own data by adjusting the sample code provided in the notebooks and by taking a closer look at the actual function implementations in all files of the ```src```-folder, in particular within ```src/utils.py``` and ```src/model.py```.

---

### Data

The whole original data set used in the paper, which includes the smart meter data of multiple heat pumps, cannot be shared. However, some examples of single heat pumps with 15-minute resolution are provided in the ```data/```-folder. This data is used in the notebook that can be found in the ```notebooks```-folder. Please further note that the weather data of the nearest weather station of each household is also already added as a separate column, next to the energy consumption. A single file in ```data/customer_data/``` refers to one household, named after its corresponding ```customer_id```.

---

### Pre-Trained Models

All models trained and evaluated in the paper are provided in the ```models``` folder. The file ```models/model_mapping.csv``` serves as an overview of the different types of models that are referred to in the paper. Note that for each type of model, five different version are provided within the subfolders, referring to the different test folds from the cross validation. When applying the models to your own data, you can choose whatever fold and model type works best for your setup. In general, however, the usage should become quite clear by following the example in ```notebooks/notebooks/02_apply_pretrained_nn.ipynb```.

---

### Further Hints and Recommendations

#### Notes on Performance

As the training setup in terms of hardware is highly individualized, we removed all code that refers to hardware optimization or the usage of GPUs. Instead the current implementations use Python's internal ```multiprocessing``` package and train ```keras``` models on all CPU resources available. Therefore, please consider the code provided here as non-accelerated or speed-optimized code. We encourage you to fork the repository and make adaptions based on you own needs and individual setup.

#### Notes on the Usage of Tensorflow Metal on Apple Silicon Chips

For newer Macs with Apple Silicon Chips (e.g., M1, M2, M3), there is a particular Python package provided by Apple to accelerate the training of machine learning models with Tensorflow, which is called ```tensorflow-metal``` (see [here](https://developer.apple.com/metal/tensorflow-plugin/)). However, whenever we tested the code provided here with an ```anaconda``` environment with ```tensorflow-metal``` installed, we gained distortions in the predictions that we cannot explain. In particular, when using the notebook ```notebooks/02_apply_pretrained_nn.ipynb```, problems occurred that could easily remain unnoticed when not having a reference on what the predictions should look like without this package installed. We note that the models provided were not trained on Macs with silicon chips. However, a further in-depth analysis to investigate the causes of distortions was not performed. **For now, we clearly recommend not to install the ```tensorflow-metal``` package in the ```anaconda``` environment that you use for the purpose of running the code in this repository.**

#### Notes on Usage of Tensorboard 

You can track the training progress of the models by using ```tensorboard```. In the ```Model``` class (see ```src/model.py```), you have the option to set the ```basepath``` parameter during object creation, which defines the path to a folder where the model files and tensorboard files should be created. By default (meaning ```basepath=None```), they are written to a folder named ```results``` within this repository, which is created by the ```Handler``` (see ```src/handler.py```) when running the code. This folder is ignored for any synchronization of this repo (as defined in the ```.gitignore```-file). Below you can find an example on how to launch a tensorboard within the anaconda environment of this repo. 

1. Start the anaconda environment with ```conda activate hp_disagg```.
2. Run the notebook ```notebooks/01_train_nn.ipynb``` to create exemplary results through training. 
3. Navigate to the folder where all tensorboard files are located: ```cd <path to this repo>/results/tensorboards/```
4. Launch tensorboard: ```tensorboard --logdir .```
5. In your browser go to ```http://localhost:6006/``` 

#### Further References 

If you are working with smart meter data and/or in the context of heat pump operations, you may want to check out other work in this area, provided on [GitHub](https://github.com/tbrumue). 
