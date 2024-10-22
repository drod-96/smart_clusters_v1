
This project contains necessary data and codes to reproduce our results in our proposed **aggregation and learning framework**. 

# Table of contents

**[Project description](#project-description)**<br>
**[Requirements](#requirements)**<br>
**[ANNs training](#anns-training)**<br>
**[License](#license)**<br>


# Project description

This project contains the source codes and post-processing pipelines of the methodology and results presented in our [article](https://doi.org/10.1016/j.egyai.2024.100393). The major parts of this repository contain the following:

- **Graph generator**: a graph generator using graph-theory based processes to generate *District Heating Networks* like graphs. A paper which dives into the details of this graph generator will be presented in future conference. Graphic representation and manipulation are realized using [Networkx](https://networkx.org), a python package built for creation, manipulation and study of complex networks structures.

- **Clusters learning**: we provide in this repository the source codes to randomly select clusters of heat consumers within District Heating Networks using random-walks and the necessary machine learning pipeline codes to train different architectures of *Artificial Neural Networks* models and replace the clusters. The machine learning models are created using [Keras](https://keras.io), an open-access and straightforward to-use deep learning library built on top of *TensorFlow*. Not *defined hyperpameters* use default values provided by Keras. By default, TensorFlow performs training on gpu if available. Training processes have been realized using on the High Performance Computer [Jean Zay](http://www.idris.fr/jean-zay/).

- **Hybrid simulation results**: we also provide post-processing jupiter notebooks to retrieve the results of the hybrid simulation presented in our [paper](https://doi.org/10.1016/j.egyai.2024.100393). Physical and Hybrid modeling codes are not available in this repository as they will be made available within an *in-progress* journal paper presenting in all details the physical components. However, our paper presents the major physical equations involved in these modeling codes. Further details can be shared at request.
  

# Requirements

## Packages

To install all required packages, enter the following line commands at the project source folder

```bash
python -m pip install -r requirements.txt
``` 

## Data

Physical **simulation results**, needed for ML models training, **case study networks** topology and **selected clusters** json files can be downloaded from open-access [Mendeley Data](https://data.mendeley.com/datasets/77stj44drm/1). Simulation folder *ARTICLE_dhn_data* must be available at the project source folder for training codes to work.


# ANNs training

Selected clusters are found in *considered_clusters_1.json* and *considered_clusters_2.json* refereed as set 1 and set 2 for each of the case study network. Clusters are identified by the parent DHN (INTEGER), the set (INTEGER) and a key (STRING). Considered ANNs achitectures are [GRU, MLP, CNN] with two versions each. Further details can be found in our paper. For instance, to train a cluster denoted with key ("a") belonging to the network (DHN 1) and from the set 1, the following command may be used.

```bash
python train.py --model=gru --model_version=1 --cluster_key="a" --cluster_dhn_id=1 --cluster_set_id=1

```


# License

Data and codes in this project are protected under the [European Union Public Licence (EUPL) v1.2](https://joinup.ec.europa.eu/page/eupl-text-11-12).
For more information see [LICENSE](LICENSE).
