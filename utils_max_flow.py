from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
import random


SEED=42
SUP_TITLE_X, SUP_TITLE_HA, SUP_TITLE_FONTSIZE = 0.26, 'center', 'x-large'
SUB_PLOTS_FIGSIZE = (12, 6)

heuristic_targets_df = pd.DataFrame()
heuristic_type = None

def get_heuristic_targets():
    global heuristic_targets_df
    return heuristic_targets_df


###------------------------------------------------------------FROM HERE N-k CAPACITY ROBUSTNESS------------------------------------------------------------###

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

                                # Weighted Flow Capacity Rate
                                if flow > 0 and heuristic_type == 'wfcr':
                                    heuristic_count[(u, v)] = heuristic_count.get((u, v), 0) + (flow ** 2) / G.edges[(u, v)]['max_cap_M_m3_per_d']
                                    continue
                                
                                # Max flow edge count
                                if (u, v) not in heuristic_count:
                                    heuristic_count[(u, v)] = 0
                                
                                # Max flow
                                if heuristic_type == 'max_flow':
                                    heuristic_count[(u, v)] += flow
                                    continue

                                # Max flow edge count
                                if flow > 0 and heuristic_type == 'max_flow_edge_count':
                                    heuristic_count[(u, v)] += 1
                                    continue

                                # Load rate
                                if flow > 0 and heuristic_type == 'load_rate':
                                    heuristic_count[(u, v)] += (flow / G.edges[(u, v)]['max_cap_M_m3_per_d'])
            else:
                flow_matrix[i, j] = 0    

    # Before returning the flow matrix, set the heuristic results
    global heuristic_targets_df
    n = G.number_of_nodes()
    try:
        heuristic_targets_df = pd.DataFrame([{'edge': k, 'wfcr': v / (n * (n - 1) * tot_flow),} for k, v in heuristic_count.items()])
        heuristic_targets_df = heuristic_targets_df.sort_values(by='wfcr', ascending=False) 
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
        
        elif heuristic == 'load_rate':
            target_df = get_heuristic_targets()
            if target_df.empty:
                return results_df
            G, flow_matrix, node_indices = perform_targeted_removal(G, 'load_rate', target_df.iloc[0].edge, flow_matrix, node_indices, results_df)
        

        elif heuristic == 'max_flow_edge_count':
            target_df = get_heuristic_targets()
            if target_df.empty:
                return results_df    
            G, flow_matrix, node_indices = perform_targeted_removal(G, 'max_flow_edge_count', target_df.iloc[0].edge, flow_matrix, node_indices, results_df)

        elif heuristic == 'max_flow':
            target_df = get_heuristic_targets()
            if target_df.empty:
                return results_df
            G, flow_matrix, node_indices = perform_targeted_removal(G, 'max_flow_edge_flows', target_df.iloc[0].edge, flow_matrix, node_indices, results_df)   

        elif heuristic == 'wfcr':
            target_df = get_heuristic_targets()
            if target_df.empty:
                return results_df
            G, flow_matrix, node_indices = perform_targeted_removal(G, 'wfcr', target_df.iloc[0].edge, flow_matrix, node_indices, results_df)

        elif heuristic == 'greedy_centrality':
            try:
                target = greedy_centrality_lst[k]
            except IndexError:
                return results_df
            
            G, flow_matrix, node_indices = perform_targeted_removal(G, 'greedy_centrality', target, flow_matrix, node_indices, results_df)
            
        else:
            raise ValueError("Invalid heuristic")


    return results_df


#------------------------------------------------------------FROM HERE ONWARDS ARE FUNCTIONS FOR PLOTTING------------------------------------------------------------

def plot_biplot(results_df, title_prefix=""):

    heuristic = str(results_df.iloc[1]['heuristic'])
    remove = 'edge' if isinstance(results_df.iloc[1]['removed_entity'], tuple) else 'node'

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=SUB_PLOTS_FIGSIZE)

    # Plot max_flow value versus k iterations
    ax1.plot(results_df.index, results_df['max_flow_value'], marker='o')
    ax1.set_xlabel('k iterations')
    ax1.set_ylabel('max_flow value')
    ax1.set_title('Max Flow Value vs k ' + heuristic + ' ' + remove + ' removals')

    # Plot capacity_robustness versus k iterations
    ax2.plot(results_df.index, results_df['capacity_robustness_max_flow'], marker='o')
    ax2.set_xlabel('k iterations')
    ax2.set_ylabel('capacity_robustness')
    ax2.set_title('Capacity Robustness vs k ' + heuristic + ' ' + remove + ' removals')

    plt.suptitle(title_prefix, x=SUP_TITLE_X, ha=SUP_TITLE_HA, fontsize=SUP_TITLE_FONTSIZE)
    plt.tight_layout()
    plt.show()

def plot_heuristic_comparison_biplot(df_list):
    fig, ax = plt.subplots()  

    series_name = df_list[0].columns[:2].tolist()[1]  
    shortest_df_length = min(len(df) for df in df_list) + 50

    for df in df_list:
        heuristic = str(df.iloc[1]['heuristic']).replace('_', ' ')
        if heuristic == 'max flow edge count':
            heuristic = 'Edge count'
        if heuristic =='max flow edge flows':
            heuristic = 'FC'
        if heuristic == 'load rate':
            heuristic = 'FCR'
        if heuristic == 'wfcr':
            heuristic = 'WFCR'
        ax.plot(df.index[:shortest_df_length], df[series_name][:shortest_df_length], label=f'{heuristic}')

    remove = 'edge' if isinstance(df_list[0].iloc[1]['removed_entity'], set) else 'node'


    ax.set_xlabel('k '+remove+' removals')
    ax.set_ylabel(series_name.replace('_', ' ')) 
    ax.legend()

    # plt.title('N-k max flow, '+ remove + ' removals', x=0.2, ha='center', fontsize=12) 
    plt.tight_layout()
    plt.show()
    return fig



#------------------------------------------------------------FROM HERE ONWARDS ARE FUNCTIONS FOR RESULTS ANALYSIS------------------------------------------------------------


def results_summary(df_, metric='', abs_or_pct='abs'):

    df = df_.copy()

    heuristic = 'entity criticality index'
    if 'max_flow_value' in df.columns and metric=='':
        metric = 'max_flow_value'
        heuristic = df.iloc[1]['heuristic']

    zero_metric_iteration = df[df[metric] == 0].index.min()

    
    # Calculate differences between consecutive rows for the metric
    df['diff'] = round(df[metric].diff() * (-1), 2) 
    df['pct_change'] = round((df['diff'] / df[metric].shift(1))*-100, 1)
    df['it'] = df.index 

    if heuristic:
        print(f"Heuristic: {heuristic}")

    print()

    print("First entity removals:")
    print('----------------------------------------------')
    print(df[['it', 'removed_entity', 'diff', 'pct_change']].iloc[1:6].to_string(index=False))

    print()

    df = df.nlargest(5, 'diff')
    if abs_or_pct == 'pct':
        df = df.nlargest(5, 'pct_diff')

    print(f"Entity removals causing most damage, measured by: {metric}")
    print('----------------------------------------------')
    print(df[['it', 'removed_entity', 'diff', 'pct_change']].to_string(index=False))

    print()
    print()

    print("Summary statistics (first 150 removals)")
    print('----------------------------------------------')

    # Calculate the percentage loss of total max flow based on first and last row
    initial_max_flow = df_.loc[0, metric]
    final_max_flow = df_.loc[df_.index[-1], metric]
    print(f"Percentage network damage: {round(((initial_max_flow - final_max_flow) / initial_max_flow) * 100, 1)}%")
    print(f"Mean damage per entity removal: {round(df.head(250)['diff'].mean(), 2)}")
    print(f"Variation in damage per entity removal: {round(df.head(250)['diff'].std(), 2)}")
    if not pd.isna(zero_metric_iteration):
        print(f"The metric reaches 0 at iteration {zero_metric_iteration}.")


    


