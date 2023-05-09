
An extensive literature review by us on BNNs have shown that, in suitably defined large data limit, **BNN's posteriors are robust to gradient-based adversarial
attacks**.Thus, this study aims to demonstrate the theoretical robustness of Bayesian
neural architectures against multiple white-box attacks and list empirical findings
from the same.

The robustness was found on 4 state of the art attack strategies listed below:

```math
l_{\infty}-FGSM,\ l_{\infty}-PGD,\ l_{2}-PGD,\ BIM
```


### Bayesian Layer types

This repository contains two types of bayesian layer implementation:  
* **BBB (Bayes by Backprop):**  
  Based on [this paper](https://arxiv.org/abs/1505.05424). This layer samples all the weights individually and then combines them with the inputs to compute a sample from the activations.

* **BBB_LRT (Bayes by Backprop w/ Local Reparametrization Trick):**  
  This layer combines Bayes by Backprop with local reparametrization trick from [this paper](https://arxiv.org/abs/1506.02557). This trick makes it possible to directly sample from the distribution over activations.

#### Bayesian
Run the file below to train the bayesian models:

`python main_bayesian.py --net_type <network> --dataset <dataset>`
* set hyperparameters in `config_bayesian.py`


#### Normal
Run the file below to train the normal models:

`python main_normal.py --net_type <network> --dataset <dataset>`
* set hyperparameters in `config_normal.py`


### Directory Structure:
`layers/`:  Contains `ModuleWrapper`, `FlattenLayer`, `BBBLinear` and `BBBConv2d`.  
`Models/BayesianModels/`: Contains standard Bayesian models (BBBLeNet, BBBAlexNet, BBB3Conv3FC,BBBResnet34,BBBVGG11).  
`Models/NonBayesianModels/`: Contains standard Non-Bayesian models (LeNet, AlexNet,Resnet34,VGG11).  
`checkpoints/`: Checkpoint directory: Models will be saved here.   
`main_bayesian.py`: Train and Evaluate Bayesian models.  
`config_bayesian.py`: Hyperparameters for `main_bayesian` file.  
`main_normal.py`: Train and Evaluate non-Bayesian (normal) models.  
`config_normal.py`: Hyperparameters for `main_normal` file.  
`AttackingModels.ipynb` : Attacks performed on BNNs and plot generated

A detailed Report can be found [here](https://drive.google.com/drive/folders/17GqazlG-ehRf07Rp7qaNRWPvUHAYHblE?usp=share_link) which was written as a part of course project for CSL7640 Dependable AI course at IIT Jodhpur.

