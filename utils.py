from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
import random

#------------------------------------------------------------PRE-PROCESSING------------------------------------------------------------

import json
import ast
from mpl_toolkits.basemap import Basemap

def expand_dict_column(row):
    # Initialize an empty dict to hold the expanded data
    expanded_data = {}
    for key, value in row.items():
        if isinstance(value, list):
            # Convert list to a string representation
            expanded_data[key] = str(value)
        else:
            # Directly assign non-list values
            expanded_data[key] = value
    return pd.Series(expanded_data)

def split_column_to_multiple(df, column, prefix=None):
    df[column] = df[column].apply(lambda x: ast.literal_eval(x))
    expanded_columns = df[column].apply(expand_dict_column).apply(pd.Series)
    if prefix:
        expanded_columns = expanded_columns.add_prefix(prefix)
    df = df.join(expanded_columns)
    df.drop(column, axis=1, inplace=True)
    return df

def split_col_to_and_from(df, col, type):
    try:
        df[col] = df[col].apply(lambda x: json.loads(x.replace("'", '"')))
        if type == 'str':
            df[col] = df[col].apply(lambda x: [str(i) for i in x])
        elif type == 'int':
            df[col] = df[col].apply(lambda x: [int(i) for i in x])
        elif type == 'float':
            df[col] = df[col].apply(lambda x: [float(i) for i in x])
        df['from_' + col] = df[col].apply(lambda x: x[0])
        df['to_' + col] = df[col].apply(lambda x: x[1])
        df = df.drop(columns=[col])
    except Exception as e:
        print(f"An error occurred while processing column '{col}': {str(e)}")
    return df

def split_coords (df, col):
    # The 'lat' and 'long' columns both on the format "[60.54204007494405, 60.54266109350547, 60.54330065370974, 60.56023487203963]"
    # We want to split this into four columns: 'from_lat', 'from_long', 'to_lat', 'to_long'
    # The first value in each column is the value of the start node, from_lat and from_long, and the last value is the end node, to_lat and to_long
    try:
        df[col] = df[col].apply(lambda x: json.loads(x.replace("'", '"')))
        df['from_' + col] = df[col].apply(lambda x: x[0])
        df['to_' + col] = df[col].apply(lambda x: x[-1])
        df = df.drop(columns=[col])
    except Exception as e:
        print(f"An error occurred while processing column '{col}': {str(e)}")
    return df

def plot_node(node_ids, graph):
    fig, ax = plt.subplots(figsize=(16, 16))

    # Use BaseMap to underlay a map of Europe
    m = Basemap(projection='merc', llcrnrlat=30, urcrnrlat=70, llcrnrlon=-15, urcrnrlon=80, resolution='l')
    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgray')

    # Plot the node
    for node_id in node_ids:
        node = graph.nodes[node_id]
        x, y = m(node['long'], node['lat'])
        m.plot(x, y, 'ro', markersize=5)

    plt.show()



#------------------------------------------------------------CENTRALITY-BASED APPROACH------------------------------------------------------------

def n_minus_k(G_, heuristic, remove='node', n_benchmarks=20, k_removals=2500):

    G = G_.copy()

    results_df = pd.DataFrame(columns=['iteration', 'removed_entity', 'NCPI', 'connectedness', 'reach', 'connectivity', 'heuristic'])

    def assess_grid_connectedness_init(G):

        weakly_connected = list(nx.weakly_connected_components(G))
        largest_component = max(weakly_connected, key=len)
        strongly_connected = list(nx.strongly_connected_components(G))
 
        diameter = 0
        if len(weakly_connected) > 1:
            for ele in weakly_connected:
                ele_comp = G.subgraph(ele)
                dia_ele_comp = nx.diameter(ele_comp.to_undirected())
                if dia_ele_comp == 0:
                    continue
                adj_dia_ele_comp = (len(ele) / dia_ele_comp) * (len(ele) / len(G.nodes))
                diameter += adj_dia_ele_comp
        else:
            diameter = G.number_of_nodes() / nx.diameter(G.to_undirected())

        CCI_dict = CCI(G.subgraph(largest_component))
        node_composite_centrality = np.average(np.array(list(dict(CCI_dict).values())))
        next_target_node = max(CCI_dict, key=CCI_dict.get)

        return len(largest_component), len(weakly_connected), len(strongly_connected), diameter, node_composite_centrality, next_target_node
    
   
    r_graphs = [G.copy() for _ in range(n_benchmarks)]
    lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init, next_target_node = assess_grid_connectedness_init(G)
    
    for i in tqdm(range(0, k_removals + 1), desc='N-k iterations'):  
        r_connectedness_lst, r_robustness_lst, r_reach_lst, r_connectivity_lst = [], [], [], []

        # For the first iteration, calculate and store initial connectedness indices
        if i == 0:  
            composite, robustness, reach, connectivity, next_target_node = NCPI(G, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init)

            results_df.loc[i] = [i, None, composite, robustness, reach, connectivity, None]
            continue

            
        # Rest of iterations (i.e., not i == 0)
        if heuristic == 'random':

            for real_copy in r_graphs:  # Iterate through the instantiated copies
                target = random.choice(list(real_copy.nodes() if remove == 'node' else real_copy.edges()))

                real_copy.remove_node(target) if remove == 'node' else real_copy.remove_edge(*target)
                r_composite, r_robustness, r_reach, r_connectivity, _ = NCPI(real_copy, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init)
                
                # Log indices for each modified graph
                r_connectedness_lst.append(r_composite)
                r_robustness_lst.append(r_robustness)
                r_reach_lst.append(r_reach)
                r_connectivity_lst.append(r_connectivity)
            
            
            results_df.loc[i] = [i, None,
                                sum(r_connectedness_lst) / len(r_connectedness_lst), sum(r_robustness_lst) / len(r_robustness_lst), sum(r_reach_lst) / len(r_reach_lst), sum(r_connectivity_lst) / len(r_connectivity_lst), 'random']

        elif heuristic == 'greedy':

            target = CCI_e(G, type='Li et al., 2021') if remove == 'edge' else next_target_node

            G.remove_node(target) if remove == 'node' else G.remove_edge(*target)
            target = target if remove == 'node' else set(target)                
            composite, robustness, reach, connectivity, next_target_node = NCPI(G, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init)

            results_df.loc[i] = [i, target, composite, robustness, reach, connectivity, 'greedy']

            if composite == 0:
                return results_df            

    return results_df

def NCPI(G, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centr_G_init):
    largest_component_size, num_weakly_connected, num_strongly_connected, diameter, node_composite_centrality, next_target_node = get_connectedness_metrics_of(G)

    """
    CONNECTEDNESS 
    Measures the network's resilience against random failures or targeted attacks. 
    It's approximated by the relative sizes of the largest connected component, weakly connected components, and strongly connected components.

    """

        
    NCPI_connectedness = (largest_component_size / lcs_G_init) * (nwc_G_init / num_weakly_connected) * ( nsc_G_init / num_strongly_connected)

    """
    REACH 
    Quantifies the efficiency of information flow or reachability across the network. 
    It's reflected in the ratio of current average shortest path length and network diameter in comparison to their initial values.
    """
    NCPI_reach = (diameter / dia_G_init)

    """
    CONNECTIVITY 
    Represents the centrality or average connectivity of the network. 
    It's estimated by comparing the current median node degree to the initial median node degree.
    """
    NCPI_connectivity = (node_composite_centrality / comp_centr_G_init)

    """
    Calculate the composite connectedness index using the above calculated indices
    """
    NCPI = NCPI_connectedness * NCPI_reach * NCPI_connectivity
    
    return NCPI, NCPI_connectedness, NCPI_reach, NCPI_connectivity, next_target_node

def CCI(G):
    """
    Returns a dictionary of composite centrality values for each node in the given graph G
    """
    node_degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(G)
    closeness = nx.closeness_centrality(G)
    epsilon = 1e-6
    composite_centralities = {
        node: (node_degree.get(node, epsilon) * betweenness.get(node, epsilon) * closeness.get(node, epsilon))**(1/3) for node in G.nodes()
    }
    return composite_centralities

def CCI_v(G):
    """ 
    Returns the node (country) with the highest composite centrality value in the given graph G
    """
    composite_centralities = CCI(G)
    return max(composite_centralities, key=composite_centralities.get)

def CCI_e(G, type=''):
    """
    Returns the edge with the highest composite centrality value in the given graph G
    """

    if type == '':
        centrality = CCI(G)
        edge_centrality = {
            edge: (centrality[edge[0]] + centrality[edge[1]]) / 2.0
            for edge in G.edges()
        }
        return max(edge_centrality, key=edge_centrality.get)
    
    if type == 'Li et al., 2021':

        edge_centrality = nx.edge_betweenness_centrality(G)
        edge_centrality = {edge: centrality for edge, centrality in edge_centrality.items() if 'super_source' not in edge and 'super_sink' not in edge}
        return max(edge_centrality, key=edge_centrality.get)
    

def get_connectedness_metrics_of(G):
    """
    Helper function that returns the following connectedness metrics of the given graph G:
    - Largest component size
    - Number of weakly connected components
    - Number of strongly connected components
    - Diameter 
    - Node composite centrality
    """
    weakly_connected = list(nx.weakly_connected_components(G))
    largest_component = max(weakly_connected, key=len, default=[])
    largest_component_size = len(largest_component)
    num_weakly_connected = len(weakly_connected)
    num_strongly_connected = len(list(nx.strongly_connected_components(G)))    

    diameter = 0
    if len(weakly_connected) > 1:
        for ele in weakly_connected:
            ele_comp = G.subgraph(ele)
            dia_ele_comp = nx.diameter(ele_comp.to_undirected())
            if dia_ele_comp == 0:
                continue
            adj_dia_ele_comp = (len(ele) / dia_ele_comp) * (len(ele) / len(G.nodes))
            diameter += adj_dia_ele_comp
    else:
        diameter = G.number_of_nodes() / nx.diameter(G.to_undirected())

    CCI_dict = CCI(G.subgraph(largest_component))
    node_composite_centrality = np.average(np.array(list(dict(CCI_dict).values())))
    next_target_node = max(CCI_dict, key=CCI_dict.get)

    return largest_component_size, num_weakly_connected, num_strongly_connected, diameter, node_composite_centrality, next_target_node
    
    








#----------------------------------------------------------MAX FLOW-BASED APPROACH------------------------------------------------------------


heuristic_targets_df = pd.DataFrame()
heuristic_type = None

def get_heuristic_targets():
    global heuristic_targets_df
    return heuristic_targets_df


def W(G, global_nodes_lst, global_sources_lst, global_sinks_lst):
    """
    Computes all-pairs flow matrix W of the network.
    
    Parameters:
        G: A NetworkX MultiDiGraph

    Returns:
        flow_matrix: 2D numpy array representing the flow matrix
        node_indices: Dictionary mapping nodes to their corresponding indices
    """
    num_nodes = len(global_nodes_lst)
    node_indices = {node: i for i, node in enumerate(global_nodes_lst)}
    flow_matrix = np.zeros((num_nodes, num_nodes))

    tot_flow = 0
    heuristic_count = {}

    for i in tqdm(range(num_nodes), desc="Computing flow matrix W"):

        source = global_nodes_lst[i]    

        if source in G and G.in_degree(source) > 0:
            for j in range(num_nodes):
                flow_matrix[i, j] = 0  
            continue  

        if source in G and source not in global_sources_lst:
            for j in range(num_nodes):
                flow_matrix[i, j] = 0  
            continue    

        for j in range(num_nodes):

            sink = global_nodes_lst[j]

            if source != sink and source in global_sources_lst and sink in global_sinks_lst and source in G and sink in G:
                if nx.has_path(G, source, sink):
                    flow_val, flow_dict = nx.maximum_flow(G, source, sink, capacity='max_cap_M_m3_per_d', flow_func=nx.algorithms.flow.dinitz)
                    flow_matrix[i, j] = flow_val
                    tot_flow += flow_val
                
                    for u, flows in flow_dict.items():
                            for v, flow in flows.items():
                                
                                if (u, v) not in heuristic_count:
                                    heuristic_count[(u, v)] = 0
                                
                                # flow centrality
                                if flow > 0 and heuristic_type == 'fc':
                                    heuristic_count[(u, v)] += flow
                                    continue

                                # flow capacity rate
                                if flow > 0 and heuristic_type == 'fcr':
                                    heuristic_count[(u, v)] += (flow / G.edges[(u, v)]['max_cap_M_m3_per_d'])

                                # weighted flow capacity rate
                                if flow > 0 and heuristic_type == 'wfcr':
                                    heuristic_count[(u, v)] = heuristic_count.get((u, v), 0) + (flow ** 2) / G.edges[(u, v)]['max_cap_M_m3_per_d']
                                    continue

            else:
                flow_matrix[i, j] = 0    

    # Before returning the flow matrix, set the heuristic results
    global heuristic_targets_df
    n = G.number_of_nodes()
    try:
        heuristic_targets_df = pd.DataFrame([{'edge': k, 'heuristic': v / (n * (n - 1) * tot_flow),} for k, v in heuristic_count.items()])
        heuristic_targets_df = heuristic_targets_df.sort_values(by='heuristic', ascending=False) 
    except KeyError:
        heuristic_targets_df = pd.DataFrame()

    return flow_matrix, node_indices, tot_flow


def W_c(_flow_matrix, target, node_indices):
    """
    Computes the flow matrix W_c after removing a node.
    Defined in Cai et al. (2021) as the original flow matrix of the network after removing entry corresponding to the removed node.

    Parameters:
        flow_matrix: Flow matrix of the original graph
        target: Target can be either a single node or an edge in the form (v1, v2)
        node_indices: Dictionary mapping nodes to their indices in the flow matrix

    Returns:
        flow_matrix_c: Flow matrix after removing the specified node
        flow_matrix: Modified flow matrix
    """

    flow_matrix = _flow_matrix.copy()

    if isinstance(target, (set,tuple)) and len(target) == 2:
        # Target is an edge in the form (v1, v2)
        v1, v2 = target
        index_v1 = node_indices.get(v1, None)
        index_v2 = node_indices.get(v2, None)

        if index_v1 is not None and index_v2 is not None:
            flow_matrix[index_v1, index_v2] = 0
            flow_matrix[index_v2, index_v1] = 0
    
    else:
        removed_node_index = node_indices.get(target, None)

        if removed_node_index is not None and removed_node_index < flow_matrix.shape[0]:
            flow_matrix = np.delete(flow_matrix, removed_node_index, axis=0)
            flow_matrix = np.delete(flow_matrix, removed_node_index, axis=1)

    return flow_matrix


def flow_capacity_robustness(G_, heuristic='random', remove='node', k_removals=2500, n_benchmarks = 10, greedy_centrality_lst=None):
    """ 
    Computes the n-k capacity robustness based on maximum flow of a graph
    """

    global heuristic_type
    heuristic_type = heuristic

    # Make a copy of the graph
    G = G_.copy()
    
    # Instantiate list of all nodes in the graph
    global_nodes_lst = list(G.nodes())

    # Define the sinks and sources configuration
    global_sources_lst = [n for n, d in G.nodes(data=True) if d.get('flow_type') == 'source']
    global_sinks_lst = [n for n, d in G.nodes(data=True) if d.get('flow_type') == 'sink']
   
    # Get all-pairs flow matrix W of the network
    flow_matrix, node_indices, flow_val_init = W(G, global_nodes_lst, global_sources_lst, global_sinks_lst)

    # Instantiate the results dataframe
    results_df = pd.DataFrame(columns=['max_flow_value', 'capacity_robustness_max_flow', 'heuristic', 'removed_entity'])
    results_df.loc[0] = [flow_val_init, 1, None, None]


    # Helper function to perform a targeted removal   
    def perform_targeted_removal(G, heuristic, target, flow_matrix, _node_indices, results_df):
        
        if remove == 'edge':
            G.remove_edge(*target)
        else:
            target = target[0] if heuristic != 'greedy_centrality' else target
            G.remove_node(target)

        # Calculate the flow matrix W_c after removing the node or edge
        W_c_ = W_c(flow_matrix, target, _node_indices)

        W_c_prime, node_indices, current_flow_val = W(G, global_nodes_lst, global_sources_lst, global_sinks_lst)

        target = target if remove == 'node' else set(target)

        results_df.loc[k] = [current_flow_val, np.sum(W_c_prime) / np.sum(W_c_), heuristic, target]

        return G, W_c_, node_indices
    
    # Initialize copies for random heuristic
    if heuristic == 'random':
        G_lst = [G.copy() for _ in range(n_benchmarks)]
        G_node_indices_lst = [node_indices.copy() for _ in range(n_benchmarks)]
        G_flow_matrix_lst = [flow_matrix for _ in range(n_benchmarks)]

    # N-k capacity robustness calculation
    for k in tqdm(range(1, k_removals + 1), desc='N-k capacity robustness'):

        if heuristic == 'random':

            print('On iteration', k)

            max_flow_lst, capacity_robustness_lst = [], []

            for G_copy, G_flow_matrix, G_node_indices in zip(G_lst, G_flow_matrix_lst, G_node_indices_lst):

                # Get a random target to remove
                target = random.choice([t for t in (G_copy.nodes() if remove == 'node' else G_copy.edges())])
                G_copy.remove_edge(*target) if remove == 'edge' else G_copy.remove_node(target)
                
                # Calculate W_c and W_c_prime after removing the node or edge
                G_flow_matrix = W_c(G_flow_matrix, target, G_node_indices)
                G_W_c_prime, G_node_indices, current_flow_val = W(G_copy, global_nodes_lst, global_sources_lst, global_sinks_lst)

                # Append the results to the lists for the current iteration
                capacity_robustness_lst.append(np.sum(G_W_c_prime) / np.sum(G_flow_matrix))
                max_flow_lst.append(current_flow_val)
            
            target = target if remove == 'node' else set(target)
            results_df.loc[k] = [np.mean(max_flow_lst), np.mean(capacity_robustness_lst), 'random', target]

        elif heuristic == 'fc':
            target_df = get_heuristic_targets()
            if target_df.empty:
                return results_df
            G, flow_matrix, node_indices = perform_targeted_removal(G, 'fc', target_df.iloc[0].edge, flow_matrix, node_indices, results_df)  

        elif heuristic == 'fcr':
            target_df = get_heuristic_targets()
            if target_df.empty:
                return results_df
            G, flow_matrix, node_indices = perform_targeted_removal(G, 'fcr', target_df.iloc[0].edge, flow_matrix, node_indices, results_df) 

        elif heuristic == 'wfcr':
            target_df = get_heuristic_targets()
            if target_df.empty:
                return results_df
            G, flow_matrix, node_indices = perform_targeted_removal(G, 'wfcr', target_df.iloc[0].edge, flow_matrix, node_indices, results_df)
            
        else:
            raise ValueError("Invalid heuristic")


    return results_df
