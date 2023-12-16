import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import mat73 as mat

CP = 4200.0
RHO = 996.0

class DistrictHeatingNetworkFromExcel(object):
    """
        This object represents a District Heating Network and associated graphic representation
    """

    def __init__(self,
                 topology_file_path: str,
                 dataset_file_path: str,                    
                 undirected = False,                             # whether to represent the DHN in directed or undirected graph
                 index_start_for_Text = 10,                     # first static steps considered (i.e., first outdoor temperatures)
                 index_end_for_Text = 4000,                     # last static steps considered 
                 last_limit_simulation = 20,                    # do not take last dynamic values due to physical model results
                 transient_delay_in_time_step = 40,):            # do not take first dynamic values due to initialization of DHN temperatures which may not be physical
    
        self.topology_file_path = topology_file_path            # topology file
        self.dataset_file_path = dataset_file_path              # simulation results .mat file
        self.undirected = undirected
        self.last_limit_simulation = last_limit_simulation
        self.adjacency_matrix = []                              # adjacency matrix of the graphic reprsentatino
        self.edges_nodes = {}                                   # list of edges containing start and end nodes, following supply pipes directions
        self.nodes_attributes = []                              # nodes attributes
        self.nodes_coordinates = {}                             # nodes 2D space coordinates
        self.edges_features = {}                                 # edges features (h, l, d, nominal_mwf)
        
        self.nodes_colors = []                                  # colors used to differentiate source and consumer nodes
        self.producer_nodes = []                                # source nodes
        self.outdoor_temperatures = {}                          # outdoor temperatures
        
        self.transient_delay = transient_delay_in_time_step
        self.index_start_for_Text = index_start_for_Text
        self.index_end_for_Text = index_end_for_Text+1
        
        self.edges_labels = {}                                   # edges labels (useful only for the graphic drawing process)
        self.nodes_labels = {}                                   # node labels (useful only for the graphic drawing process)
        self.inverted_edges_indices = []                        # contains the indices (in 0 base) of the edges with inverted fluxes
        
        # Generate adjacency matrix from topology file
        self._generate_adjacency_matrix_and_nodes_attributes()
        
        # Generate the dictionary of the physical results containing the temperatures of waters at (nodes and pipes), mass flow rates, demands, etc.
        self.dict_physical_values = self._generate_input_dictionnary()
    
    def get_inverted_edges_in_julia_indices(self):
        """Gets the list of edges with inverted fluxes at base 1 (for julia)

        Returns:
            list[int]: list of indices
        """
        return [i+1 for i in self.inverted_edges_indices]

    def _compute_delay_time_pipe(self, ms, l, d):
        """Computes delay time (l must be in meter)

        Args:
            ms: mass flow rates
            l: pipe length
            d: pipe diameter
        """
         # tau = (r^2 * pi * L * rho) / m [R.Hagg]
        r = d / 2
        return r**2 * np.pi * l * RHO / ms

    def _generate_adjacency_matrix_and_nodes_attributes(self):
        """Generates the adjacency matrix in the shape of (N x N) where adj[i, j] = np.array([tau/tau_p])

        Returns:
            tuple: adj or adj, coordinates, edges list or adj, edges list or adj, coordinates
        """

        nodes_df = pd.read_excel(self.topology_file_path, sheet_name=['nodes'])['nodes']
        pipes_df = pd.read_excel(self.topology_file_path, sheet_name=['pipes'])['pipes']
        loads_df = pd.read_excel(self.topology_file_path, sheet_name=['loads'])['loads']
        consumers_df = pd.read_excel(self.topology_file_path, sheet_name=['consumers'])['consumers']
        outdoor_temps_df = pd.read_excel(self.topology_file_path, sheet_name=['outdoor temperature'])['outdoor temperature']
        
        self.outdoor_temperatures = outdoor_temps_df.iloc[range(self.index_start_for_Text+self.transient_delay, self.index_end_for_Text - self.last_limit_simulation)] # as I cut in 30 points in DHN
        
        n = len(nodes_df)
        # Adjacency matrix
        adj_matrix = np.zeros(shape=(n, n), dtype=float)
        edge_index = 0
        # Attributes matrix
        att_matrix = np.zeros(shape=(n, 3), dtype=float)
        
        for edge_index, row in pipes_df.iterrows():
            start_node_index = int(row['start node']) -1
            end_node_index = int(row['end node'])-1
            self.edges_nodes[edge_index] = (start_node_index, end_node_index)
            self.edges_features[edge_index] = {
                'h': row['h'],
                'l': row['length'] * 1e3, # to meter
                'd': row['Diameter'], # Total diameter with insulation (in meter)
            }

        for node_index, row in nodes_df.iterrows():
            x = row['x']
            y = row['y']
            demand = np.mean(loads_df[node_index+1])
            is_prod = row['is_prod']
            cns = consumers_df.iloc[node_index]
            u_factor = cns['U factor']
            # tr = row['Return temperature']
            # te = row['Outside temperature']
            
            att_matrix[node_index] = np.array([is_prod, demand, u_factor], dtype=float)
            
            self.nodes_coordinates[node_index] = np.array([x, y], dtype=float)
            
            if is_prod:
                self.nodes_colors.append('tab:red')
                self.producer_nodes.append(node_index)
            else:
                self.nodes_colors.append('tab:blue')
                
            self.nodes_labels[node_index] = f'[{round(demand/1e3)}]'
            
        self.adjacency_matrix = adj_matrix
        self.nodes_attributes = att_matrix
        
        if not self.undirected:
            G = nx.from_numpy_array(self.adjacency_matrix, create_using=nx.MultiDiGraph)
        else:
            G = nx.from_numpy_array(self.adjacency_matrix)
        
        self.Graph = G
    
    def plot_district_heating_graph(self):
        """Plots the DH of study in graph form
        
        Returns:
            None
        """            
        keys = list(self.nodes_coordinates.keys())
        
        labels_to_use = self.nodes_labels
        plt.figure(figsize=(18, 18))
        nx.draw_networkx_labels(self.Graph, pos=self.nodes_coordinates, labels=labels_to_use, horizontalalignment="left", verticalalignment='bottom', font_weight=2)
        nx.draw_networkx_edge_labels(self.Graph, pos=self.nodes_coordinates, edge_labels=self.edge_labels)
        nx.draw(self.Graph, pos=self.nodes_coordinates, node_color=self.nodes_colors)
        plt.show()
        
    def _generate_input_dictionnary(self) -> dict:
        """Generates the input dictionnary
            
            Some of the edges have been arbitrary chosen to be in opposite so we have "full" negative flow rate, 
            we must change this orientation by multipliying by - 1 the mass rate

        Returns:
            dict: the dictionnary containing the values
        """
        mat_file = self.dataset_file_path
        cut = self.transient_delay
        limit = self.last_limit_simulation
        
        matlab_variables = mat.loadmat(mat_file, use_attrdict=True)
        
        # Demands loads 
        loads = matlab_variables['load'][:,cut:-limit]

        # Supply temperature at the center of each node (combination from all suppliers) = temperature supplied to next nodes
        nodes_supply_temperature_dynamic = matlab_variables['ts'][:,cut:-limit]
        # Return temperature at the center of each node (combination from all returners) = temperature returned to previous nodes
        nodes_return_temperature_dynamic = matlab_variables['tr'][:,cut:-limit]
        
        # Temperatures at the exit of the pipes
        pipes_tsin = matlab_variables['tsin'][:,cut:-limit]
        pipes_tsout = matlab_variables['tsout'][:,cut:-limit]
        pipes_trin = matlab_variables['trin'][:,cut:-limit]
        pipes_trout = matlab_variables['trout'][:,cut:-limit]
        mass_rates_in_pipes = matlab_variables['mw'][:,cut:-limit]
        consumptions_ms_rates = matlab_variables['mc'][:,cut:-limit]
        charges_at_nodes = np.zeros_like(loads)
        
        taus = []
        # Mass rates
        for edge_idx in self.edges_nodes:
            msw = mass_rates_in_pipes[edge_idx, :]
            (st, ed) = self.edges_nodes[edge_idx]
            row = self.edges_features[edge_idx]
            tau_edge = self._compute_delay_time_pipe(np.mean(np.abs(msw)), row['l'], row['d'])
            taus.append(tau_edge)
            nb_ = ed
            if all(item <= 0.1 for item in msw):
                # topologically inverted
                mass_rates_in_pipes[edge_idx, :] = np.abs(msw)
                # inverse nodes
                print(f'Edge {edge_idx+1} is inverted due to topology consideration!')
                self.edges_nodes[edge_idx] = (ed, st)
                nb_ = st
                self.inverted_edges_indices.append(edge_idx)

            charges_at_nodes[nb_,:] = mass_rates_in_pipes[edge_idx, :] * CP * (nodes_supply_temperature_dynamic[nb_,:] - nodes_return_temperature_dynamic[nb_,:])
            # CHANGER ICI SI JE VEUX CONSIDERER DES GRAPHES ORIENTES
            # self.adjacency_matrix[st, ed] = self._compute_weight_for_gnn(np.mean(np.abs(msw)), row['length']*1e3, row['Diameter'], row['h'])
            self.adjacency_matrix[st, ed] = 1
            self.adjacency_matrix[ed, st] = -1 # directed graph
            self.edges_labels[(st, ed)] = round(self.adjacency_matrix[st, ed],2)
            
        # self.edges_features.insert(loc=len(self.edges_features.columns), column='Delay time', value=np.array(taus))
        
        dict_inputs = {
            'Demands': np.abs(loads), # car en kW
            'mw': mass_rates_in_pipes,
            'Tr_node': nodes_return_temperature_dynamic,
            'Ts_node': nodes_supply_temperature_dynamic,
            'Chr_node': charges_at_nodes,
            'Trc': matlab_variables['trcs'],
            'Tsin': pipes_tsin,
            'Tsout': pipes_tsout,
            'Trin': pipes_trin,
            'Trout': pipes_trout,
            'mc': consumptions_ms_rates
        }
        
        return dict_inputs
      
    def generate_sequential_input_data_v5(self, cluster_of_nodes: list[int]) -> tuple:
        """Generates the dataset to train ML models to replace the cluster

        Args:
            cluster_of_nodes (list[int]): list of nodes composing the clusters

        Returns:
            tuple: (dataframe table of inputs, dataframe table of outputs)
        """
        
        dict_inputs = self.dict_physical_values
        _, in_going_edges, out_going_edges = self.identify_cluster_edges(cluster_of_nodes) # at time = 0
        
        demands = dict_inputs['Demands'].copy()
        ms_rates = dict_inputs['mw'].copy()
        pipes_tsin = dict_inputs['Tsin'].copy()
        pipes_tsout = dict_inputs['Tsout'].copy()
        pipes_trin = dict_inputs['Trin'].copy()
        pipes_trout = dict_inputs['Trout'].copy()
        
        df_input_features = pd.DataFrame()
        iii = 0
            
        df_outputs = pd.DataFrame()
        ooo = 0
        
        for el in cluster_of_nodes:
            node_index = int(el)-1
            demands_node = demands[node_index,:].T
            df_input_features.insert(loc=iii, column=f'Demand {node_index+1}', value=demands_node)
            iii += 1
        
        # incoming supply pipes
        for edge_idx in in_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edges_features[edge_idx]
            h = row['h']
            lng = row['l'] 
            d = row['d']

            tsin_edge = pipes_tsin[edge_idx,:]
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            
            weight = np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:].T))
            
            ts_in = tsin_edge.T
            df_input_features.insert(loc=iii, column=f'Tsin_pipe{edge_idx}->{sti+1}', value=ts_in*weight)
            iii +=1
            
            tr_out = trout_edge.T  
            df_outputs.insert(loc=ooo, column=f'Trout_pipe{edge_idx}->{sti+1}', value=tr_out)
            ooo += 1
        
        # outgoing supply pipes
        for edge_idx in out_going_edges:
            (sti, endi) = self.edges_nodes[edge_idx]
            row = self.edges_features[edge_idx]
            h = row['h']
            lng = row['l']
            d = row['d']
            
            tsin_edge = pipes_tsin[edge_idx,:]
            tsout_edge = pipes_tsout[edge_idx,:]
            trin_edge = pipes_trin[edge_idx,:]
            trout_edge = pipes_trout[edge_idx,:]
            
            weight = np.exp(-1*(h*lng*d/CP/ms_rates[edge_idx,:].T))
            
            ts_out = tsout_edge.T
            df_outputs.insert(loc=ooo, column=f'Tsout_pipe{edge_idx}->{endi+1}', value=ts_out)
            ooo += 1

            tr_in = trin_edge.T
            df_input_features.insert(loc=iii, column=f'Trin_pipe{edge_idx}->{endi+1}', value=tr_in*weight)
            iii += 1
        
        return df_input_features, df_outputs

    def identify_cluster_edges(self, nodes_in_cluster: list[str]) -> tuple:
        """Identify internal and external edges of the cluster inside the DHN within their directions

        Args:
            nodes_in_cluster (list[str]): list of nodes composing the clusters
            
        Returns:
            tuple: internal edges, incoming edges, outgoing edges
        """
        internal_edges_idx = []
        external_edges_idx_pointing_in = []
        external_edges_idx_pointing_out = []
        
        # on cherche dans la dict
        for edge_idx in self.edges_nodes:
            (st_n, en_n) = self.edges_nodes[edge_idx]
            if (st_n+1) in nodes_in_cluster:
                if (en_n+1) in nodes_in_cluster:
                    # ca veut dire que c'est une connexion interne
                    internal_edges_idx.append(edge_idx)
                else:
                    # ca veut dire que c'est une connexion externe
                    external_edges_idx_pointing_out.append(edge_idx)
            elif (en_n+1) in nodes_in_cluster and (st_n+1) not in nodes_in_cluster:
                external_edges_idx_pointing_in.append(edge_idx)
          
        # Pour eviter les repetitions on prend un set, similaire à HashSet pour C#
        inner_edges = set(internal_edges_idx)
        in_going_edges = set(external_edges_idx_pointing_in)
        out_going_edges = set(external_edges_idx_pointing_out)
        
        return inner_edges, in_going_edges, out_going_edges