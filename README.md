
This project contains necessary data and codes to reproduce our results in our proposed **aggregation and learning framework**. 

# Table of contents

**[Project description](#project-description)**<br>
**[Requirements](#requirements)**<br>
**[ANNs training](#anns-training)**<br>
**[License](#license)**<br>


# Project description

This project contains codes to create graphic representation of simulated District Heating Networks, to generate random-walk based clusters, and machine learning pipeline framework to learn and replace cmlusters. 

Graphic representation and manipulation are realized using [Networkx](https://networkx.org), a python package built for creation, manipulation and study of complex networks structures.

The machine learning models are created using [Keras](https://keras.io), an open-access and straightforward to-use deep learning library built on top of tensorflow. Not *defined hyperpameters* use default values provided by Keras. By default, tensorflow performs training on gpu if available.


# Requirements

## Packages

To install all required packages, enter the following line commands at the project source folder

```bash
python -m pip install -r requirements.txt
``` 

## Data

Physical **simulation results**, needed for ML models training, **case study networks** topology and **selected clusters** json files can be downloaded from open-access [drive project](https://drive.google.com/drive/folders/1JOSh7wHtEryVk4NW7ptaDTdXBQCPYDtb?usp=drive_link). Simulation folder *ARTICLE_dhn_data* must be available at the project source folder for training codes to work.


# ANNs training

Selected clusters are found in *considered_clusters_1.json* and *considered_clusters_2.json* refereed as set 1 and set 2 for each of the case study network. Clusters are identified by the parent DHN (INTEGER), the set (INTEGER) and a key (STRING). Considered ANNs achitectures are [GRU, MLP, CNN] with two versions each. Further details can be found in our paper. For instance, to train a cluster denoted with key ("a") belonging to the network (DHN 1) and from the set 1, the following command may be used.

```bash
python train.py --model=gru --model_version=1 --cluster_key="a" --cluster_dhn_id=1 --cluster_set_id=1

```


# License

Data and codes in this project are protected under the [European Union Public Licence (EUPL) v1.2](https://joinup.ec.europa.eu/page/eupl-text-11-12).
For more information see [LICENSE](LICENSE).