import matplotlib.pyplot as plt
from random import random
import networkx as nx
import numpy as np
from itertools import product
import pandas as pd
import os

LOWER_CONV_COEFF = 0.2
UPPER_CONV_COEFF = 1.2 # Le modele physique converge pas quand cette perte est trop grande

LOWER_DIAMETER = 0.01 # 1 cm
UPPER_DIAMETER = 0.1 # 10 cm

RECENT_BUILDING_U = 0.84 # W.K^1.m^-2
ANCIENT_BUILDING_U = 3.4
OFFICE_BUILDING_U = 2.5

class GraphGeneratorParameters():
    def __init__(self, 
                 E_cp = 0.5, # Probability of having central producer
                 E_rp = 0.2, # Probability of having producer per region
                 E_ee = 0.1, # Probability of having edges between regions
                 chp_pwr = 50e6,
                 nb_nodes_per_region = 100,
                 nb_regions = 4,
                 start_node_num = 0,
                 max_degree = 3,
                 jump_node_num = 1, 
                 max_diameter = 12,
                 min_cycle_length = 3,
                 target_ratio = 1.01,
                 min_distance_bt_producers = 6, 
                 edge_weight_mean = 1.5,
                 edge_weight_std = 0.2):
        self.E_central_producer = E_cp
        self.E_region_produce = E_rp
        self.E_bt_regions_pipes = E_ee
        self.max_degree = max_degree
        self.jump_node_numm = jump_node_num
        self.max_diameter = max_diameter
        self.start_node_num = start_node_num
        self.min_cycle_length = min_cycle_length
        self.nb_nodes_per_region = nb_nodes_per_region
        self.nb_regions = nb_regions
        self.target_ratio = target_ratio
        self.min_distance_bt_producers = min_distance_bt_producers
        self.edge_weight_mean = edge_weight_mean
        self.edge_weight_std = edge_weight_std
        
        self.CHP_power = chp_pwr
        

class GraphDHNGenerator(object):
    def __init__(self, control_params: GraphGeneratorParameters = None, remove_short_cycles=True):
        if control_params == None:
            control_params = GraphGeneratorParameters()
        self.params = control_params
        self.graph = nx.DiGraph()
        self.producer_indices = []
        self.node_colors = []
        self.node_indices = []
        self.remove_short_cycles = remove_short_cycles
        self.node_positions = []
        self.reload_graph = False
        self.dhn = None
        
    def _generate_random_weight(self):
        mean = self.params.edge_weight_mean
        std = self.params.edge_weight_std
        weight = np.random.default_rng().normal(mean, std)
        if weight <= 0:
            weight = self.params.edge_weight_mean / 2.0
        return weight
        
    def _show_short_cycles(self, G:nx.Graph):
        short_cycles = []
        if self.remove_short_cycles:
            try:
                cycles = {}
                for n in list(G.nodes()):
                    cl = nx.find_cycle(G, source=n)
                    key = f'{cl[0][0]}_{cl[0][1]}'
                    cycles[key] = cl
                    
                for cl in cycles:
                    cycle_tuple_list = list(cycles[cl])
                    if len(cycle_tuple_list) <= self.params.min_cycle_length:
                        short_cycles.append(cycle_tuple_list)
            except Exception:
                pass
        if len(short_cycles) != 0:
            print('Short cycles found: ')
            for cl in short_cycles:
                print(cl)
        return G

    def _remove_self_loop(self, G: nx.Graph()):
        for (u, v) in iter(nx.selfloop_edges(G)):
            G.remove_edge(u, v)
        return G

    def _generate_random_graph(self, G, remaining_nodes, jump_node_num, max_degree, node_num):
        if not remaining_nodes:
            return G
        current_node = node_num
        for _ in range(np.random.randint(1, max_degree)):
            try:
                neighbor = np.random.choice(remaining_nodes)
                G.add_edge(current_node, neighbor, weight=self._generate_random_weight())
                remaining_nodes.remove(neighbor)
            except Exception:
                break
        
        n = node_num + jump_node_num
        self._generate_random_graph(G, remaining_nodes, jump_node_num, max_degree, n)

    def _generate_region_graph(self, center, n):
        max_degree = self.params.max_degree
        jump_nd = self.params.jump_node_numm
        max_diameter = self.params.max_diameter
        
        G = nx.Graph()
        remaining_nodes = list(range(n))
        # Generate the random graph
        self._generate_random_graph(G, remaining_nodes, jump_nd, max_degree, 0)
        
        while not nx.is_connected(G):
            node1 = np.random.choice(list(G.nodes()))
            node2 = np.random.choice(list(G.nodes()))
            if G.degree(node1) >= max_degree or G.degree(node2) >= max_degree or node1 == node2:
                continue
            else:
                G.add_edge(node1, node2)

        for e in iter(nx.selfloop_edges(G)):
            G.remove_edge(e[0], e[1])

        # Ensure that the graph has the desired diameter
        it = 0
        while nx.diameter(G) < max_diameter and it < 100:
            node1 = np.random.choice(list(G.nodes()))
            node2 = np.random.choice(list(G.nodes()))
            G.add_edge(node1, node2)
            it+=1
        
        pos = nx.kamada_kawai_layout(G, center=center)
        return G, pos

    def generate_random_region_with_target(self, center):
        params = self.params
        nb = params.nb_nodes_per_region
        ratio = 100
        G = nx.Graph()
        # if nb < 60:
        #     print('Minimumn number of nodes is 60')
        #     nb = 60
        
        G, pos = self._generate_region_graph(center, nb)
        ratio = G.number_of_edges() / (G.number_of_nodes() - 1)
        while np.abs(ratio - params.target_ratio) > 1e-1:
            if ratio > params.target_ratio:
                # recreer un autre graph
                G, pos = self._generate_region_graph(center, nb)
            else:
                node1 = np.random.choice(list(G.nodes()))
                node2 = np.random.choice([item for item in list(G.nodes()) if item != node1])
                G.add_edge(node1, node2)
            G = self._remove_self_loop(G)
            ratio = G.number_of_edges() / (G.number_of_nodes() -1)
        # print(f'Ratio (E/V) = {ratio}')
        return G
    
    def generate_random_dhn(self):
        params = self.params
        center_coordinates_of_trees = [[-1, -1], [1, 1], [-1, 1],[2, 1],[1,2]]
        
        ## OUTPUT Varibales
        node_colors = [] # Color of nodes
        labels = {} # Label of nodes
        positions = [] # Spatial coordinates of nodes
        producers = [] # Producer units indices
        
        initial_pos = 0
        labels_per_graphs = {}    
        
        ## GENERATION PROCESS
        if len(node_colors) != 0:
            node_colors = [] # Color of nodes
            labels = {} # Label of nodes
            positions = [] # Spatial coordinates of nodes
            producers = [] # Producer units indices
            labels_per_graphs = {}

        add_central_chp_producer = random() < params.E_central_producer # Un producteur central

        # Main graph for DHN
        u_graph = nx.Graph()

        if add_central_chp_producer:
            labels[0] = str(0)
            u_graph.add_node(0)
            positions.append(np.array([np.random.uniform(-1,2), np.random.uniform(-1,2)], dtype=float))
            node_colors.append('tab:red')
            producers.append(0)
            
        count_graph = 0
        ii = 0
        print('Generating each region ...')
        while count_graph < params.nb_regions:
            print(f'\tRegion {count_graph+1}')
            if count_graph == len(center_coordinates_of_trees):
                ii = 0
            center_position_of_tree = center_coordinates_of_trees[ii]
            ii+=1
            last_nd = u_graph.number_of_nodes()
            div_coordinate = np.array(center_position_of_tree, dtype=float)

            G = self.generate_random_region_with_target(div_coordinate)
            pos = nx.kamada_kawai_layout(G, center=2*div_coordinate, scale=2)
            
            starting_label = last_nd
            # print(f'Starting label = {starting_label}')
            # Probability of having producer
            if not add_central_chp_producer and count_graph == 0:
                e_rb = 1
            else:
                e_rb = params.E_region_produce

            if np.random.uniform(0, 1) < e_rb:
                producer_idx = np.random.randint(starting_label, starting_label + G.number_of_nodes()) # Chose one to be supplier
                producers.append(producer_idx)
            
            for n in pos:
                positions.append(pos[n])
                node_lab = n + starting_label
                labels[node_lab] = str(node_lab)
                if count_graph not in labels_per_graphs:
                    labels_per_graphs[count_graph] = []
                labels_per_graphs[count_graph].append(node_lab)
                
                if node_lab in producers:
                    node_colors.append('tab:red')
                else:
                    node_colors.append('tab:blue')
            
            u_graph = nx.disjoint_union(u_graph, G)
            if add_central_chp_producer:
                ts = np.random.choice(list(G.nodes()))
                u_graph.add_edge(0, int(ts)+starting_label, weight=self._generate_random_weight())
            else:
                if count_graph != 0:
                    id_othr_graph = np.random.choice(labels_per_graphs[count_graph])
                    if count_graph == 1:
                        id_prev_graph = np.random.choice(labels_per_graphs[0])
                    else:
                        id_prev_graph = np.random.choice(labels_per_graphs[np.random.randint(0,count_graph-1)])
                    u_graph.add_edge(id_othr_graph, id_prev_graph, weight=self._generate_random_weight())  
            
            count_graph += 1  
        
        it = 0
        looping = False
        print('Loop --> adding edges between regions')
        while not looping and it < 10:
            # Random edges connections
            for gi in range(count_graph-1): # Count - 1 = number total
                for gj in range(gi+1, count_graph-1):
                    if np.random.uniform() <= params.E_bt_regions_pipes:
                        u = np.random.choice(labels_per_graphs[gi])
                        v = np.random.choice(labels_per_graphs[gj])
                        if u != v and not u_graph.has_edge(u, v):
                            u_graph.add_edge(u, v)
            looping = nx.is_connected(u_graph)
            it+=1

        print('\t --> finished')
        poss = nx.kamada_kawai_layout(u_graph, scale=10, dim=2, weight='weight')
        if nx.is_connected(u_graph):
            close_producers = []
            for i in range(len(producers)):
                prd = producers[i]
                for j in range(i+1, len(producers)):
                    prdx = producers[j]
                    if nx.shortest_path_length(u_graph, prd, prdx) < params.min_distance_bt_producers:
                        if [prd, prdx] in close_producers or [prdx, prd] in close_producers:
                            continue
                        close_producers.append([prd, prdx])
            
            for [id1, id2] in close_producers:
                node_colors[id2] = 'tab:blue'
                if id2 in producers:
                    producers.remove(id2)
            print('Close producers = ',close_producers)
            print('Diameter = ',nx.diameter(u_graph))
            print('Ratio (E/V) total =',u_graph.number_of_edges() / (u_graph.number_of_nodes() -1))
            self._show_short_cycles(u_graph)
            self.graph = u_graph
            self.node_colors = node_colors
            self.node_indices = labels
            self.node_positions = poss
            self.producer_indices = producers
            print('Graph generation done !!')
            self.plot_district_heating_network()
            return True
        else:
            print('WARNING!!!! NOT CONNECTED GRAPH !! GENERATE AGAIN')
            return False
    
    def read_generated_graph(self, excel_file_topology: str, plot_graph=True):
        # utiliser "expand graph" pour visualiser graph
        excel_data = pd.read_excel(excel_file_topology, sheet_name=['nodes', 'pipes'])
        df_nodes = excel_data['nodes']
        df_pipes = excel_data['pipes']
        n_nodes = len(df_nodes)
        n_pipes = len(df_pipes)
        node_colors = []
        labels = {}
        producer_indices = []
        positions = np.zeros(shape=(n_nodes, 2), dtype=float)
        adjacency_matrix = np.zeros(shape=(n_nodes, n_nodes), dtype=int)
        for n in range(n_nodes):
            positions[n] = np.array([df_nodes.iloc[n]['x'],df_nodes.iloc[n]['y']])
            labels[n] = n
            if df_nodes.iloc[n]['is_prod'] == 1:
                node_colors.append('tab:red')
                producer_indices.append(n)
            else:
                node_colors.append('tab:blue')
            
        for indx, row in df_pipes.iterrows():
            st = int(row['start node']) - 1
            ed = int(row['end node']) - 1
            adjacency_matrix[st, ed] = 1
            adjacency_matrix[ed, st] = 1

        self.graph = nx.from_numpy_array(adjacency_matrix)
        self.producer_indices = producer_indices
        self.node_colors = node_colors
        self.node_indices = labels
        self.node_positions = positions
        
        if plot_graph:
            self.plot_district_heating_network()
    
    def plot_district_heating_network(self):
        plt.figure(figsize=(42,40))
        nx.draw(self.graph, pos=self.node_positions, labels=self.node_indices, node_color=self.node_colors, node_size=200, font_size=12, font_color="black")
        plt.show()
    
    def generate_random_connected_dhn(self):
        is_generated = False
        while not is_generated:
            is_generated = self.generate_random_dhn()
    
    def assign_values_create_topology_file(self, graph_name:str):
        # u_graph.add_edges_from([(u, v) for u, v in product(labels_per_graphs[0], labels_per_graphs[1]) if random() < probability_of_having_pipes_between_trees])  
        folder = 'Plausible_dhns'
        if not os.path.exists(folder):
            os.mkdir(folder)
        labels = [int(item) +1 for item in self.node_indices]
        sorted_labels = labels.copy()
        sorted_labels.sort()
        G = self.graph
        positions = self.node_positions
        nodes_metadata = []
        nodes_informations = []
        for n in sorted_labels:
            i = n -1
            is_prod = 1 if i in self.producer_indices else 0 
            nodes_metadata.append([n, positions[i][0], positions[i][1], is_prod])
        df_nodes_metadata = pd.DataFrame(nodes_metadata, columns=['nbr','x','y','is_prod'])
        
        pipes_metadata = []
        for pipe in G.edges():
            (start_node, end_node) = pipe
            length = np.linalg.norm(np.array(positions[start_node]) - np.array(positions[end_node]))
            ls = np.random.uniform(LOWER_CONV_COEFF, UPPER_CONV_COEFF)
            dia = np.random.uniform(LOWER_DIAMETER, UPPER_DIAMETER)
            pipes_metadata.append([int(start_node)+1, int(end_node)+1, dia, ls, length])
        df_pipes_metadata = pd.DataFrame(pipes_metadata, columns=['start node', 'end node', 'Diameter', 'h', 'length'])
        
        cons_metadata = []
        df_loads = pd.DataFrame()
        df_nantes_data = pd.read_excel(os.path.join(folder, 'Nantes_Power_load_data.xlsx'), sheet_name=['Data'])['Data']
        temps = np.array(df_nantes_data['T2M'], dtype=float)
        hours = np.array(df_nantes_data['HOUR_STEP'], dtype=int)
        df_loads.insert(loc=0, column='hours', value=hours)
        loads = []
        for n in sorted_labels:
            loads_n = np.ones_like(temps)
            area = 0
            u = 0
            gen_f = 0
            is_space_heating = 0
            is_industrial_heating = 0
            
            if np.random.randint(0,10) < 5:
                is_space_heating = 1
                id = np.random.choice([0, 1, 2])
                area = np.random.uniform(5,12) # Entre 5000-12000m2 de surface d'echange
                if id == 0: # ancient building
                    u = np.random.default_rng().normal(ANCIENT_BUILDING_U, 0.1)
                elif id == 1:
                    u = np.random.default_rng().normal(RECENT_BUILDING_U, 0.1)
                else:
                    u = np.random.default_rng().normal(OFFICE_BUILDING_U, 0.1)
                peak = u*area*1e3*(18+6)
                for t in range(len(loads_n)):
                    if temps[t] >= 18.0:
                        loads_n[t] = 0.2*peak
                    else:
                        sanitary_shape = np.random.uniform(2,5)*0.1
                        loads_n[t] = u*area*1e3*np.abs(18.0-temps[t]) + sanitary_shape*peak    
                
            else:
                is_industrial_heating = 1
                gen_f = np.random.choice([100,200,300,400])
                loads_n = gen_f*(np.sin((np.array(temps)/10) - 24 ) + 2)
            
            cons_metadata.append([n, area, u, gen_f, is_space_heating, is_industrial_heating])
            loads.append(loads_n)
        
        df_cons = pd.DataFrame(cons_metadata, columns=['nbr', 'surface area', 'U factor', 'Gen-factor', 'Space heating', 'Industrial use'])
        nn_loads = pd.DataFrame(np.array(loads).T, columns=sorted_labels)
        df_loads = pd.concat([df_loads, nn_loads], axis=1)
        nx.write_gml(G, os.path.join(folder, graph_name))
        
        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter(os.path.join(folder, graph_name+'_topology.xlsx'), engine='xlsxwriter')
        
        df_nodes_metadata.to_excel(writer, sheet_name='nodes', index=False, index_label=False)
        df_pipes_metadata.to_excel(writer, sheet_name='pipes', index=False, index_label=False)
        df_nantes_data.to_excel(writer, sheet_name='outdoor temperature', index=False, index_label=False)
        df_cons.to_excel(writer, sheet_name='consumers', index=False, index_label=False)
        df_loads.to_excel(writer, sheet_name='loads', index=False, index_label=False)
        writer.close()
        print(f'Graph {graph_name} saved!')