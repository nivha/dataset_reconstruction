<h1 align="center"> Reconstructing Training Data <br> from Trained Neural Networks </h1>

<h3 align="center"> 
<a href="https://nivha.github.io/" target="_blank">Niv Haim</a>*, 
<a href="https://scholar.google.co.il/citations?user=LVk3xE4AAAAJ&hl=en" target="_blank">Gal Vardi</a>*,
<a href="https://scholar.google.co.il/citations?user=opVT1qkAAAAJ&hl=iw" target="_blank">Gilad Yehudai</a>*,
<a href="https://www.wisdom.weizmann.ac.il/~shamiro/" target="_blank">Ohad Shamir</a>,
<a href="https://www.weizmann.ac.il/math/irani/" target="_blank">Michal Irani</a>
</h3>

<h4 align="center"> NeurIPS 2022 (Oral) </h4>

<h3 align="center"> 
<a href="https://giladude1.github.io/reconstruction" target="_blank">Webpage</a>, 
<a href="https://arxiv.org/abs/2206.07758" target="_blank">Paper</a>
</h3>

Pytorch implementation of the NeurIPS 2022 paper: [Reconstructing Training Data from Trained Neural Networks](https://arxiv.org/abs/2206.07758).

#### 

## Setup

Create a copy of ```setting.default.py``` with the name ```setting.py```, and make sure to change the paths inside to match your system. 

Create and initialize new conda environment using the supplied ```environment.yml``` file (using python 3.8 and pytorch 11.1.0 and CUDA 11.3) :
```
conda env create -f environment.yaml
conda activate rec
```


## Running the Code

### Notebooks
For quick access, start by running the provided notebooks for analysing the (already provided) 
reconstructions of two (provided) models for binary CIFAR10 (vehicles/animals) and MNIST (odd/even):

- ```reconstruction_cifar10.ipynb```
- ```reconstruction_mnist.ipynb```


### Reproducing the provided trained models and their reconstructions

All training/reconstructions are done by running ```Main.py``` with the right parameters.  
Inside ```command_line_args``` directory we provide command-lines with necessary arguments 
for reproducing the training of the provided models and their provided reconstructions
(those that are analyzed in the notebooks)  


#### Training
For reproducing the training of the provided two trained MLP models (with architecture D-1000-1000-1):

 - CIFAR10 model (for reproduction run ```command_line_args/train_cifar10_vehicles_animals.txt```)
 - MNIST model (for reproduction run ```command_line_args/train_mnist_odd_even_args.txt```)

#### Reconstructions

In ```reconstructions``` directory we provide two reconstructions (results of two runs) per each of the models above.

To find the right hyperparameters for reconstructing samples from the above models 
(or any other models in our paper) we used Weights & Biases sweeps.
In general, it is still an open question how to find the right hyperparameters 
for our losses without trial and error.

These reconstructions can be reproduced by running the following commandlines (the right hyperparameter can be found there):

- CIFAR10: ```command_line_args/reconstruct_cifar10_b9dfyspx_args.txt``` and ```command_line_args/reconstruct_cifar10_k60fvjdy_args.txt```
- MNIST: ```command_line_args/reconstruct_mnist_kcf9bhbi_args.txt``` and ```command_line_args/reconstruct_mnist_rbijxft7_args.txt```

(For full reconstruction, one has to run a hyperparameter sweep. We used W&B to do this. An exapmle for a W&B sweep specs can be found [here](https://github.com/nivha/dataset_reconstruction/issues/2]))


### Training/Reconstructing New Learning Problems

One should be able to train/reconstruct models for new problems by adding a 
new python file under ```problems``` directory.

Each problem file should contain the logic of how to load the data and 
the parameters necessary to build the model. 


## BibTeX

```bib
@inproceedings{haim2022reconstructing,
 author = {Haim, Niv and Vardi, Gal and Yehudai, Gilad and Shamir, Ohad and Irani, Michal},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {S. Koyejo and S. Mohamed and A. Agarwal and D. Belgrave and K. Cho and A. Oh},
 pages = {22911--22924},
 publisher = {Curran Associates, Inc.},
 title = {Reconstructing Training Data From Trained Neural Networks},
 url = {https://proceedings.neurips.cc/paper_files/paper/2022/file/906927370cbeb537781100623cca6fa6-Paper-Conference.pdf},
 volume = {35},
 year = {2022}
}
```
