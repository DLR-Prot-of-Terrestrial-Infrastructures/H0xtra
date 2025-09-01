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

### Pretrained Models

To only replicate the results from our paper, you can use the provided weights for pretrained models. Choose the task (i.e., 'nextloc' or 'trajgen'), the model version (i.e., 'zs' for zero-shot, 'sc' for single-city, or 'mc' for multi-city) and the city to test on. For instance, evaluating the zero-shot trajectory generation on city C can be done by executing:

`python testing.py --task 'trajgen' --model 'zs' --test_city '3'`

All results are stored in **results**.


### Results

Below, the results for next-location prediction and trajectory generation are provided. Each block represents the city the models are tested on. H0xtra refers to our zero-shot model, trained on all cities except the test city. H0xtra-SC refers to our single-city model that is trained and tested on the same city. H0xtra-MC refers to our multi-city model that is trained on all four cities.

#### Next-Location Prediction

|     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
| Metric | Top-1 Acc. (↑) | Top-5 Acc. (↑) | Top-10 Acc. (↑) | Top-20 Acc. (↑) | MRR (↑) | Distance Error (↓) |
| **City A** |     |      |     |     |     |     |
| H0xtra | 0.257 | 0.520 | 0.602 | 0.670 | 0.377 | 2351.7 |
| H0xtra-SC | 0.315 | 0.612 | 0.686 | 0.745 | 0.450 | 2001.1 |
| H0xtra-MC | 0.329 | 0.629 | 0.695 | 0.750 | 0.466 | 1982.0 |
| **City B** |     |     |     |     |     |     |
| H0xtra | 0.315 | 0.612 | 0.680 | 0.742 | 0.453 | 2213.6 |
| H0xtra-SC | 0.280 | 0.574 | 0.668 | 0.745 | 0.414 | 2257.3 |
| H0xtra-MC | 0.330 | 0.640 | 0.712 | 0.772 | 0.471 | 2127.7 |
| **City C** |     |     |     |     |     |     |
| H0xtra | 0.297 | 0.579 | 0.646 | 0.703 | 0.426 | 2044.1 |
| H0xtra-SC | 0.279 | 0.580 | 0.664 | 0.727 | 0.416 | 2023.3 |
| H0xtra-MC | 0.314 | 0.616 | 0.682 | 0.734 | 0.454 | 1953.3 |
| **City D** |     |     |     |     |     |     |
| H0xtra | 0.273 | 0.566 | 0.649 | 0.714 | 0.406 | 2088.4 |
| H0xtra-SC | 0.219 | 0.498 | 0.594 | 0.682 | 0.347 | 2382.9 |
| H0xtra-MC | 0.281 | 0.589 | 0.670 | 0.735 | 0.420 | 2062.0 |

#### Trajectory Generation

|     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- |
| Metric (JSD) (↓) | Distance | Radius | Duration | Periodicity | I-rank | G-rank |
| **City A** |     |     |     |     |     |     |
| H0xtra | 0.0846 | 0.0770 | 0.0524 | 0.4282 | 0.0177 | 0.0855 |
| H0xtra-SC | 0.0007 | 0.0018 | 0.0002 | 0.0127 | 0.0000 | 0.0717 |
| H0xtra-MC | 0.0018 | 0.0016 | 0.0011 | 0.0356 | 0.0347 | 0.1003 |
| **City B** |     |     |     |     |     |     |
| H0xtra | 0.0134 | 0.0125 | 0.0114 | 0.1543 | 0.0562 | 0.0261 |
| H0xtra-SC | 0.0007 | 0.0035 | 0.0007 | 0.0052 | 0.0000 | 0.0126 |
| H0xtra-MC | 0.0013 | 0.0038 | 0.0003 | 0.0194 | 0.0000 | 0.0410 |
| **City C** |     |     |     |     |     |     |
| H0xtra | 0.0328 | 0.0425 | 0.0184 | 0.2116 | 0.0523 | 0.0303 |
| H0xtra-SC | 0.0005 | 0.0074 | 0.0007 | 0.0365 | 0.0347 | 0.0371 |
| H0xtra-MC | 0.0016 | 0.0043 | 0.0005 | 0.0237 | 0.0000 | 0.0344 |
| **City D** |     |     |     |     |     |     |
| H0xtra | 0.0102 | 0.0150 | 0.0048 | 0.1253 | 0.0562 | 0.0564 |
| H0xtra-SC | 0.0007 | 0.0047 | 0.0004 | 0.0324 | 0.0347 | 0.0583 |
| H0xtra-MC | 0.0019 | 0.0023 | 0.0003 | 0.0249 | 0.0347 | 0.1281 |


### License
This repository is published under license GPL-3.0.


### Contribute
If you want to contribute, please open a new issue.