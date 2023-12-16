RANDOM_STATE_SHUFFLE = 2

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import matplotlib as mpt
from cycler import cycler
import os
import json

from src.dhnv2 import DistrictHeatingNetworkFromExcel

from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle

random.seed(RANDOM_STATE_SHUFFLE)
mpt.rcParams['lines.linewidth'] = 1.4
mpt.rcParams['font.size'] = 12
mpt.rcParams['axes.prop_cycle'] = cycler(color=['r', 'g', 'b', 'y'])
plt.style.use('default')

def get_dhn_from_id(network_id: int, folder_dhn: str) -> DistrictHeatingNetworkFromExcel:  
    """Gets the DHN case study based on the id [1,2,3,4]

    Args:
        network_id (int): the case study dhn id
        folder_dhn (str): path of folder containing .mat (simulation) and .xlsx (topology) files

    Returns:
        DistrictHeatingNetworkFromExcel: the case study DHN
    """
    
    topology_file = ""
    dataset_file = ""

    folder_dhn_id = os.path.join(folder_dhn, f"Network_{network_id}")
    for dirName, _, files_list in os.walk(folder_dhn_id):
        if len(files_list) >= 2:
            for fname in files_list:
                filename, file_ext = os.path.splitext(fname)
                if filename.startswith('dataset') and file_ext == ".mat":
                    dataset_file = os.path.join(dirName, fname)
                elif file_ext == ".xlsx":
                    topology_file = os.path.join(dirName, fname)
    
    return DistrictHeatingNetworkFromExcel(topology_file_path=topology_file,
                                           dataset_file_path=dataset_file,
                                           undirected=True, 
                                           transient_delay_in_time_step=0, 
                                           last_limit_simulation=60)

def get_dhn_clusters(dhn_id: int, clusters_json_folder: str) -> dict:
    """Loads all selected clusters sets (1 and 2) into dictionary

    Args:
        dhn_id (int): the id of the case study DHN
        clusters_json_folder (str): Path of the folder containing the case study DHNs information

    Returns:
        dict: the selected clusters with their keys
    """
    # Path sure to exist here
    dict_ = {}
    path_dhn = os.path.join(clusters_json_folder, f"Network_{dhn_id}")
    if os.path.exists(path=path_dhn):
        set_1_json_file = os.path.join(path_dhn, "considered_clusters_1.json")
        with open(set_1_json_file, mode="r") as jsonfile:
            dict_["v1"] = json.load(jsonfile)
        jsonfile.close()
        
        set_2_json_file = os.path.join(path_dhn, "considered_clusters_2.json")
        with open(set_2_json_file, mode="r") as jsonfile:
            dict_["v2"] = json.load(jsonfile)
        jsonfile.close()
            
        return dict_

def generate_reduced_network_node_id_dict(clustered_nodes: list[int]) -> dict:
    """Generates dictionary to link the non-clustered nodes ids to their original ids (in the full netwokr)
    
    Args:
        clustered_nodes (list[int]): list of clustered nodes (in julia indexing)

    Returns:
        dict: dictionary to associate the original node id to the reduced network node ids
    """
    step = 0
    dict_ = {}
    for i in range(71): # There is 71 nodes in the case study network 1
        if (i+1) in clustered_nodes:
            step += 1
        dict_[i] = i-step
    return dict_

def set_my_layout(ax: plt.Axes, xlabel: str, ylabel: str, title: str=None, legend_loc=(0.01,0.01), withlegend=False, legends=None) -> plt.Axes:
    """Sets my personal layout used in all official figures

    Args:
        ax (plt.Axes): the pyplot axes
        xlabel (str): x-axis label
        ylabel (str): y-axis label
        title (str, optional): title of the figure. Defaults to None.
        legend_loc (tuple, optional): location of the legend box in 2D coordinates. Defaults to (0.01,0.01).
        withlegend (bool, optional): whether to put the legend box. Defaults to False.
        legends (_type_, optional): legends list. Defaults to None.

    Returns:
        plt.Axes: the pyplot axes (self return)
    """
    if title != None:
        ax.set_title(title, fontdict={'color': 'black', 'size': 14})
    ax.set_xlabel(xlabel, fontdict={'color': 'black', 'size': 20})
    ax.set_ylabel(ylabel, fontdict={'color': 'black', 'size': 20})
    ax.tick_params(axis='y', colors='black', labelsize=18)
    ax.tick_params(axis='x', colors='black', labelsize=18)
    if withlegend:
        ax.legend(legends, prop={'size': 16}, loc=legend_loc)
    # ax.xaxis.major.formatter._useMathText = True
    return ax

def create_time_sequential_dataset(X, y, time_steps = 60):
    """Creates sequential dataset with length of time_steps

    Args:
        X (np.array): Input features
        y (np.array): Output features
        time_steps (int, optional): length of sequential data. Defaults to 1.

    Returns:
        (np.array, np.array): X data sequenced, Y data sequenced
    """
    xs, ys = [], []
    for i in range(len(X)-time_steps):
        v = X[i:i+time_steps, :]
        xs.append(v)
        ys.append(y[i+time_steps, :])
    return np.array(xs), np.array(ys)

def compute_confidence_interval(distribution, z=1.960):
    """Computes confidence interval

    Args:
        distribution (np.array): data
        z (float, optional): z-value. Defaults to 1.960.

    Returns:
        (np.array, confidence): (mean, +- confidence)
    """
    # 95%, z = 1.960
    # 99%, z = 2.576
    array_1d = distribution.reshape(-1, 1)
    n = len(array_1d)
    std = np.std(array_1d)
    return np.mean(array_1d), z*std / np.sqrt(n)


def step_scheduler(epoch, lr):
    """Performs step-scheduler learning rates

    Args:
        epoch (int): current training epoch
        lr (float): current learning rate

    Returns:
        float: modified learning rate
    """
    if epoch != 0 and epoch%10==0:
        lr = lr*0.1
    return lr


def get_cluster_types(dhn: DistrictHeatingNetworkFromExcel, cluster: list[str], cluster_name: str) -> dict:
    """Identifies the types of the clusters based on the outside connections

    Args:
        dhn (DistrictHeatingNetworkFromExcel): the DHN
        cluster (list[str]): the list of nodes labels in the DHN
        cluster_name (str): cluster key in the list of clusters

    Returns:
        dict: 'Text': says the type of the cluster,
                'Type': the type
    """
    _, ins, outs = dhn.identify_cluster_edges(cluster)
    return {
        'Text': f'Cluster {cluster_name} is type {len(ins)}in-{len(outs)}out',
        'Type': f'{len(ins)}-{len(outs)}' # 1-0
    }


def _get_data_from_dataframe(df_inputs:pd.DataFrame, df_outputs:pd.DataFrame, shuffle_data=False, time_step=60):
    """Gets the data from inputs and outputs dataframe tables

    Args:
        df_inputs (pd.DataFrame): inputs features table
        df_outputs (pd.DataFrame): output features table
        shuffle_data (bool, optional): Whether to shuffle train sets. Defaults to False.
        time_step (int, optional): Length of input sequences. Defaults to 60.

    Returns:
        tuple: (train input, train output, test input, test output,scaller outputs, scaller inputs)
    """
    df_inputs = df_inputs.iloc[2000:-60]
    df_outputs = df_outputs.iloc[2000:-60]
    
    inputs = df_inputs.copy()
    outputs = df_outputs.copy()
    
    X = np.asarray(inputs)
    Y = np.asarray(outputs)
    
    print('Input features: ',df_inputs.columns)
    print('Output features: ',df_outputs.columns)

    scaller_x = MinMaxScaler().fit(X)
    scaller_y = MinMaxScaler().fit(Y)

    X = scaller_x.transform(X)
    Y = scaller_y.transform(Y)

    x, y = create_time_sequential_dataset(X, Y, time_step) 
    
    if shuffle_data:
        x, y = shuffle(x, y, random_state = RANDOM_STATE_SHUFFLE)
    
    n = x.shape[0]
    num_train_samples = round(0.2*n) # 20 % test sets
    
    test_x, train_x = np.split(x, indices_or_sections=[num_train_samples], axis=0)
    test_y, train_y = np.split(y, indices_or_sections=[num_train_samples], axis=0)
    
    return train_x, train_y, test_x, test_y, scaller_y, scaller_x

def flatten_data(train_x: np.array, train_y: np.array) -> tuple:
    """Flattens the input sequencial data from large matrice data for each input features

    Args:
        train_x (np.array): sequencial input data
        train_y (np.array): output data

    Returns:
        tuple: (flattened input data, output data (not changed))
    """
    
    flat_x_train = train_x.reshape(-1, train_x.shape[1]*train_x.shape[2])
    return flat_x_train, train_y
    
def transform_time_to_time_data(train_x: np.array, train_y: np.array) -> tuple:
    """Transforms the sequencial input-output data to input-output data (time-to-time)

    Args:
        train_x (np.array): input data
        train_y (np.array): output data

    Returns:
        tuple: (input data, output data)
    """
    train_x_ = []
    train_y_ = []
    for i_data in range(train_x.shape[0]):
        train_x_.append(train_x[i_data,-1,:])
        train_y_.append(train_y[i_data,:])
    return np.array(train_x_), np.array(train_y_)

def generate_input_sequencial_data(dhn: DistrictHeatingNetworkFromExcel, cluster: list[str], shuffle_data=True, time_step=60) -> tuple:
    """Generates and performs prepocessing of input/output data features for the cluster

    Args:
        dhn (DistrictHeatingNetworkFromExcel): the DHN containing the cluster with the simulation results
        cluster (list[str]): list of nodes composing the cluster
        shuffle_data (bool, optional): whether to shuffle the data. Defaults to True.
        time_step (int, optional): the length of the sequential input features. Defaults to 60.

    Returns:
        tuple: (train input, train output, test input, test output,scaller outputs, scaller inputs)
    """
    # inners, ins, outs, qls = dhn.get_cluster_qualities_and_identify_connecting_pipes(cluster)
    df_inputs, df_outputs = dhn.generate_sequential_input_data_v5(cluster)
    train_x, train_y, test_x, test_y, scaller_y, scaller_x = _get_data_from_dataframe(df_inputs, df_outputs, shuffle_data=shuffle_data, time_step=time_step)
    return train_x, train_y, test_x, test_y, scaller_y, scaller_x


def generate_random_walk_cluster_from_dhn(dhn_network:DistrictHeatingNetworkFromExcel, control_params):
    """Generates the random-walk clusters from the DHN

    Args:
        dhn_network (DistrictHeatingNetworkFromExcel): original DHN
        control_params (dict): control parameters of the random walks (nb_of_walkers, max_nodes, list_producers)

    Raises:
        Exception: if any key of control parameters not defined

    Returns:
        dict: list of clusters generated
    """
    
    # P. Pons & M. Latapy, "Computing communities in large networks using random walks"
    # https://www-complexnetworks.lip6.fr/~latapy/Publis/communities.pdf

    # C. Toth et al. "Synwalk - Community Detection via Random Walk Modelling", 2021

    key_nb_walkers = 'nb_of_walkers'
    key_nb_max_nodes = 'max_nodes'
    key_producers_list = 'list_producers'
    
    keys = [key_nb_walkers, key_nb_max_nodes, key_producers_list]
    for key_ in keys:
        if key_ not in control_params:
            raise Exception(f'Key {key_} not in control_params, verify!')
        
    adj_matrix = dhn_network.adjacency_matrix
    adj_matrix_unweighted = np.zeros_like(adj_matrix)
    adj_matrix_unweighted[adj_matrix.nonzero()] = 1.
    
    # Degree Matrix
    degree_matrix = np.zeros_like(adj_matrix)
    for edge_key in dhn_network.edges_nodes:
        (start_node, end_node) = dhn_network.edges_nodes[edge_key]
        degree_matrix[start_node, start_node] += 1
        degree_matrix[end_node, end_node] += 1
        
    # Transition Probability matrix
    transition_probability_matrix = np.matmul(adj_matrix_unweighted,np.linalg.inv(degree_matrix))

    clusters_found = []
    limit_nb_clusters = control_params[key_nb_walkers] # == number of walkers
    limit_size_clusters = control_params[key_nb_max_nodes]
    producers_from_julia = control_params[key_producers_list]

    nb_nodes = degree_matrix.shape[0]
    list_nodes = range(nb_nodes)
    producers = [nd-1 for nd in producers_from_julia]
    while len(clusters_found) < limit_nb_clusters:
        current_cluster = []
        current_node = np.random.choice(list_nodes)
        limit_nb_steps = np.random.choice(range(2, limit_size_clusters)) # Walker steps
        for t in range(limit_nb_steps):
            prob_p = transition_probability_matrix[:,current_node]
            prob_p[producers] = 0.
            prob_p[current_cluster] = 0.0
            sum_pr = np.sum(prob_p) 
            if sum_pr == 0:
                break
            prob_p *= 1/sum_pr
            founds = random.choices(population=list_nodes, weights=prob_p)
            next_node = founds[0]
            retry = 0
            while (next_node in producers or next_node in current_cluster) and retry < 4:
                founds = random.choices(population=list_nodes, weights=prob_p)
                next_node = founds[0]
                retry += 1
            if retry == 4:
                print('Reached!')
            current_cluster.append(next_node)
            current_node = next_node
        current_cluster.sort()
        if len(current_cluster) <=1 or current_cluster in clusters_found:
            continue 
        
        clusters_found.append([ids+1 for ids in current_cluster])
            
    print(f'{len(clusters_found)} clusters selected')

    clusters_dict = {}
    clusters_dist_types = []
    clusters_dist_nb = []
    key = 0
    for cluster in clusters_found:
        cluster_key = f'{key}-c'
        type_c_str = get_cluster_types(dhn_network, cluster, cluster_key)
        type_c = type_c_str.split(' ')[-1]
        cluster_key = f'{key}-c'
        print(type_c_str + f' with {len(cluster)} nodes')
        clusters_dict[cluster_key] = cluster
        clusters_dist_types.append(type_c)
        clusters_dist_nb.append(len(cluster))
        key +=1
        
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(clusters_dist_types, bins=100)
    ax.set_xlabel('Cluster types')
    ax.set_ylabel('Distribution')
    plt.show()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(clusters_dist_nb, bins=100)
    ax.set_xlabel('Nb of nodes in the cluster')
    ax.set_ylabel('Distribution')
    plt.show()
    
    return clusters_found