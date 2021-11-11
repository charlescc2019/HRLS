# A Hierarchical Reconciliation Least Square Method for Linear Regression

#### The repository contains the experiment data and implementation of this HRLS model in [PAKDD 2022 Paper](http://pakdd.net/). 
The HRLS dataset including the simulated dataset and real dataset which have been uploaded to GitHub warehouse. You can use the create_data.py file to generate simulated dataset if you want. More details of the reproduce work are presented in the Paper to be delivered.
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




#### Pointnet2 Compile
To run the hierarchical model, one has to comile the tensorflow operations of pointnet2 (`models/sem_seg/pointnet2/tf_op`).
The compiled `*.so` files in this repo was based on CUDA 10.0 and above python packages.



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
Each folder with <area_name> contains the point cloud and label data of one area. The `h_matrices` folders contains the hierarchical linear relationship between the label in one level and the bottom level. For other structure of data, one can modify data config file `data_list.yaml` to set customized path. In addition, the train/val/test split can be reset by the data config file.

For the setting of sampling and model, each folder in `configs` contains one version of setting. The default config folder is `configs/sem_seg_default_block`, and there are captions for arguments in the config file of this folder.

To apply training of the model:
```bash
cd Campus3D
python engine/train.py -cfg <config_dir>
```
The default `<config_dir>` is `configs/sem_seg_default_block`. The model will be saved in `log/<dir_name>`, where the `<dir_name>` is the set "OUTPUT_DIR" in the config file.


To apply evaluation of the model on the test set:
```bash
cd Campus3D
python engine/eval.py -cfg  <config_dir> -s TEST_SET -ckpt <check_point_name> -o <output_log> -gpu <gpu_id>
```
The `<check_point_name>` is the name of ckpt in `log/<dir_name>`, where the `<dir_name>` is the set "OUTPUT_DIR" in the config file. The result of IoU, Overall Accuracy and Consistency Rate wiil be written into `<output_log>`, for which the default name depends on the datetime. `<gpu_id>` is to set the gpu id for 'faiss' implementation of GPU based nearest neighbour search.
