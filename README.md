# A Hierarchical Reconciliation Least Square Method for Linear Regression

#### The repository contains the experiment data and implementation of this HRLS model in [PAKDD 2022 Paper](http://pakdd.net/). 
We propose a novel hierarchical forecasting structure of linear regression model and hierarchical reconciliation least square (HRLS) method, which can improve the accuracy of forecasting and consistency of forecasting，especially when the modelling uncertainty increased. The HRLS dataset including the simulated dataset and real dataset which have been uploaded to GitHub warehouse. You can use the create_data.py file to generate your own simulated dataset if you want. More details of the reproduce work are presented in the Paper to be delivered.
![](fig2.png)

### Running Environment
#### Python packages
The repo has been tested on Python 3.9.

|  Package   | Version  |
|  ----  | ----  |
|numpy|1.21.0|
|sklearn|0.0|
|statsmodels|0.12.2|
|pandas|1.3.0|
|matplotlib|3.4.2|
|mxnet|1.7.0|




#### Create simulated  dataset
To run the hierarchical model, we  has to construct our our simulated data
To generate the dataset of the hierarchical  model:
```bash
cd Generate_SimulatedData
python create_data.py
```
You can choose different random seeds to generat different manual datasets.



### Training and Evaluation 
Download the Latest Version of HRLS and place them into `data`. The file folder should be in the following structure:
```
├── Generate_SimulatedData
│   ├── create_data.py
|   └── dataset
|       └──manual_dataAA.xlsx
|       └── ...
|   └── pic
|       └──AA.png
|       └── ...
├── RealData_Test
│   ├── k-fold cross-validation.py
|   └── dataset
|       └──IncomeOfAutomobileIndustry.xlsx
|       └── ...
├── SimulatedData_Test
│   ├── divide_dataset.py
|   ├── HRLS_Test.py
|   ├── log.txt
|   └── dataset
|       └──manual_dataAA.xlsx
|       └──manual_dataAB.xlsx
|       └── ...
```
Each folder with `dataset` contains the required data for the corresponding test program. The `pic` folders contains the hierarchical linear relationship between the label in top level and the bottom level.  In addition, the train/val/test split dataset  can be reset by the divide_dataset.py config file.

To divide the dataset of the model:
```bash
cd SimulatedData_Test
python divide_dataset.py
```

To apply training of the model:
```bash
cd SimulatedData_Test/RealData_Test
python HRLS_Test/k-fold cross-validation.py 
```
The model will be saved in `SimulatedData_Test/log.txt`, where every test result and model coefficient will store in it.

