## Official code for the ICDM 2025 paper ‘Zero-Shot Cross-City Trajectory Prediction Using Hypernetworks’
--------------------------------------------------------------------------------------------

This is the project page contains the code for the paper “Zero-Shot Cross-City Trajectory Prediction Using Hypernetworks”, developed at the Institute for the Protection of Terrestrial Infrastructure at the German Aerospace Center. If you are interested in further work, feel free to visit our [website](https://www.dlr.de/en/pi)!

The paper introduces **H0xtra**, a model for trajectory prediction (either predicting the next location of a given trajectory or generating whole trajectories from scratch) that can be directly applied to unseen cities without any retraining, i.e., performd zero-shot predictions. The general idea is as follows: using publicly available point of interest (POI) data, the model learns generalized location representations via a CNN U-Net, acting as a hypernetwork. These representations parameterize a Transformer backbone, thereby defining a space in which trajectories are disentangled from their corresponding cities of origin. This enables the Transformer to learn city-agnostic trajectory pattern. When transfering to a city unseen during training, the model only requires POI data from that target city. The CNN U-Net then produces location representations that map target city-specific trajectories into the previously learned representation space. In that space, the Transformer can apply its learned patterns to derive future locations for given trajectories or produce a large set of synthetic trajectories without any target city mobility data needed.

<figure>
    <img src="/assets/model.png"
         alt="Overview of the proposed architecture">
    <figcaption>Overview of the proposed architecture.</figcaption>
</figure>


### Repository Organization

*   **data** contains
    *   all scripts to prepare the trajectory data from raw check-in data
    *   dataset and dataloader for training and testing
*   **model** contains the code for the model architecture
*   `training.py` contains the training routine and testing
*   `metrics.py` contains the implementation of the adopted metrics for evaluation
*   **pretrained** contains weights for trained versions of 
    *   the zero-shot H0xtra in **zs.pt**
    *   the non-zero-shot H0xtra-SC in **sc.pt**
    *   the non-zero-shot H0xtra-MC in **mc.pt**
*   `testing.py` contains the code to run the experiments with the pretrained models


### Requirements
*   Pytorch > 2.0
*   Scipy
*   Numpy
*   Pandas
*   tqdm
*   matplotlib

Please install the pytorch version that suits your system. We also provide an `environment.yaml` for directly creating an environment for cuda 12.4.


### Experiments

To replicate the experiments in the paper, download the data at [https://doi.org/10.5281/zenodo.14219563](https://doi.org/10.5281/zenodo.14219563), extract all csv files into **data/raw**, and execute `data/POI_preprocessing.py` as well as `data/traj_preprocessing.py`.

For training, execute `training.py` with specifying the desired task (i.e., 'nextloc' or 'trajgen'), the source cities for training, and the cities for testing. Here, the cities A, B, C, and D are referred to as city 1, 2, 3, and 4, respectively. For instance, for the task of next-location prediction, training on cities A, B, and C, and testing on city D, execute:

`python training.py --task 'nextloc' --train_cities '[1,2,3]' --test_cities '[4]'`

#### Pretrained Models

To only replicate the results from our paper, you can use the provided weights for pretrained models. Choose the task (i.e., 'nextloc' or 'trajgen'), the model version (i.e., 'zs' for zero-shot, 'sc' for single-city, or 'mc' for multi-city) and the city to test on. For instance, evaluating the zero-shot trajectory generation on city C can be done by executing:

`python testing.py --task 'trajgen' --model 'zs' --test_city '3'`

All results are stored in **results**.


### License
This repository is published under license GPL-3.0.


### Contribute
If you want to contribute, please open a new issue.