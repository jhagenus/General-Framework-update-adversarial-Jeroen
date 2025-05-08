# Realistic Adversarial Attacks for Robustness Evaluation of Trajectory Prediction Models via Future State Perturbation
This is the used framework for generating the Adversarial attacks in the paper **Put paper reference here**. 

This is a clone of [STEP](https://github.com/DAI-Lab-HERALD/General-Framework/tree/main), a general framework for evaluating prediction models (including on perturbed data). Consequently, users who want to use our work should go there for a larger collection of models, datasets, and metrics.

The attacked model was initially trained by running [simulations_train.py](https://github.com/jhagenus/General-Framework-update-adversarial-Jeroen/blob/main/Framework/simulations_train.py). However, for the convivnience of the user, the resulting trained models (as well as the initial LGAP data set to be perturbed) are allready include in the [Results folder](https://github.com/jhagenus/General-Framework-update-adversarial-Jeroen/tree/main/Framework/Results/L-GAP%20(left%20turns)). 

To generate the resulting metric values one can run [simulations_perturbation.py](https://github.com/jhagenus/General-Framework-update-adversarial-Jeroen/blob/main/Framework/simulations_perturbation.py) in general, and [simulations_perturbation_CR_FNC.py ](https://github.com/jhagenus/General-Framework-update-adversarial-Jeroen/blob/main/Framework/simulations_perturbation_CR_FNC.py) in particular for the calculation of the $CR_{FNC}$ metric. 


