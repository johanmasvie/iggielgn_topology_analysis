from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
import random


SEED=42
SUP_TITLE_X, SUP_TITLE_HA, SUP_TITLE_FONTSIZE = 0.12, 'center', 'x-large'
SUB_PLOTS_FIGSIZE = (12, 6)

#------------------------------------------------------------FROM HERE ONWARDS IS CODE FROM PROJECT THESIS------------------------------------------------------------

def n_minus_k(G_, heuristic, remove='node', n_benchmarks=20, k_removals=2500, exclude_benchmark=True, greedy_max_flow_lst=None, SEED=42):

    G = G_.copy()

    results_df = pd.DataFrame(columns=['iteration', 'removed_entity', 'composite', 'robustness', 'reach', 'connectivity', 'composite_b', 'robustness_b', 'reach_b', 'connectivity_b'])
    results_best_worst_df = pd.DataFrame(columns=['iteration', 'best_entity', 'composite_best', 'worst_entity', 'composite_worst'])

    if exclude_benchmark:
        results_df = pd.DataFrame(columns=['iteration', 'removed_entity', 'composite', 'robustness', 'reach', 'connectivity', 'heuristic'])

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

        node_composite_centrality = np.average(np.array(list(dict(CCI(G)).values())))
        return len(largest_component), len(weakly_connected), len(strongly_connected), diameter, node_composite_centrality
    
   
    benchmark_graphs = [get_banchmark(G.to_undirected(), model='ER') for _ in range(n_benchmarks)]
    r_graphs = [G.copy() for _ in range(n_benchmarks)]

    lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init = assess_grid_connectedness_init(G)
    lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, dia_Gb_init, comp_centrs_Gb_init = assess_grid_connectedness_init(benchmark_graphs[0])
    
    for i in tqdm(range(0, k_removals + 1), desc='N-k iterations'):  # Loop through k_removals + 1 iterations
        b_connectedness_lst, b_robustness_lst, b_reach_lst, b_connectivity_lst = [], [], [], []
        r_connectedness_lst, r_robustness_lst, r_reach_lst, r_connectivity_lst = [], [], [], []

        # For the first iteration, calculate and store initial connectedness indices
        if i == 0:  
            composite, robustness, reach, connectivity = GCI(G, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init)

            if exclude_benchmark:
                results_df.loc[i] = [i, None, composite, robustness, reach, connectivity, None]
                continue

            composite_b, robustness_b, reach_b, connectivity_b = GCI(benchmark_graphs[0], lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, dia_Gb_init, comp_centrs_Gb_init)
            
            results_df.loc[i] = [i, None, composite, robustness, reach, connectivity, composite_b, robustness_b, reach_b, connectivity_b]
            results_best_worst_df.loc[i] = [i, None, composite, None, composite]
            continue  

        # Rest of iterations (i.e., not i == 0)
        if heuristic == 'random':

            for real_copy in r_graphs:  # Iterate through the instantiated copies
                target = random.choice(list(real_copy.nodes() if remove == 'node' else real_copy.edges()))

                real_copy.remove_node(target) if remove == 'node' else real_copy.remove_edge(*target)
                r_composite, r_robustness, r_reach, r_connectivity = GCI(real_copy, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init)
                
                # Log indices for each modified graph
                r_connectedness_lst.append(r_composite)
                r_robustness_lst.append(r_robustness)
                r_reach_lst.append(r_reach)
                r_connectivity_lst.append(r_connectivity)
            
            if exclude_benchmark:
                results_df.loc[i] = [i, None,
                                    sum(r_connectedness_lst) / len(r_connectedness_lst), sum(r_robustness_lst) / len(r_robustness_lst), sum(r_reach_lst) / len(r_reach_lst), sum(r_connectivity_lst) / len(r_connectivity_lst), 'random']
                continue
            
            
            for benchmark in benchmark_graphs:
                target_b = random.choice(list(benchmark.nodes() if remove == 'node' else benchmark.edges()))
                benchmark.remove_node(target_b) if remove == 'node' else benchmark.remove_edge(*target_b)
                b_composite, b_robustness, b_reach, b_connectivity = GCI(benchmark, lcs_Gb_init, nwc_Gb_init, nsc_Gb_init,  dia_Gb_init, comp_centrs_Gb_init)
                
                b_connectedness_lst.append(b_composite)
                b_robustness_lst.append(b_robustness)
                b_reach_lst.append(b_reach)
                b_connectivity_lst.append(b_connectivity)

            results_df.loc[i] = [i, None,
                                sum(r_connectedness_lst) / len(r_connectedness_lst), sum(r_robustness_lst) / len(r_robustness_lst), sum(r_reach_lst) / len(r_reach_lst), sum(r_connectivity_lst) / len(r_connectivity_lst), 
                                sum(b_connectedness_lst) / len(b_connectedness_lst), sum(b_robustness_lst) / len(b_robustness_lst), sum(b_reach_lst) / len(b_reach_lst), sum(b_connectivity_lst) / len(b_connectivity_lst)]



        elif heuristic == 'greedy':
            func = CCI_v if remove == 'node' else lambda G: CCI_e(G, type='Li et al., 2021')
            target = func(G)

            G.remove_node(target) if remove == 'node' else G.remove_edge(*target)
            target = target if remove == 'node' else set(target)                
            composite, robustness, reach, connectivity = GCI(G, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init)

            if exclude_benchmark:
                results_df.loc[i] = [i, target, composite, robustness, reach, connectivity, 'greedy']
                continue

            for benchmark in benchmark_graphs:
                benchmark_target = func(benchmark)
                benchmark.remove_node(benchmark_target) if remove == 'node' else benchmark.remove_edge(*benchmark_target)
                b_composite, b_robustness, b_reach, b_connectivity = GCI(benchmark, lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, dia_Gb_init, comp_centrs_Gb_init)
                
                b_connectedness_lst.append(b_composite)
                b_robustness_lst.append(b_robustness)
                b_reach_lst.append(b_reach)
                b_connectivity_lst.append(b_connectivity)

            results_df.loc[i] = [i, target, composite, robustness, reach, connectivity, sum(b_connectedness_lst) / len(b_connectedness_lst), sum(b_robustness_lst) / len(b_robustness_lst), sum(b_reach_lst) / len(b_reach_lst), sum(b_connectivity_lst) / len(b_connectivity_lst)]

        elif heuristic == 'max_flow':
            if greedy_max_flow_lst is None:
                raise ValueError("Please provide a list of greedy max flows for the 'max_flow' heuristic.")
            try:
                target = greedy_max_flow_lst[i]
            except IndexError:
                return results_df, results_best_worst_df
            
            if isinstance(target, tuple):
                if target not in G.edges():
                    target = tuple([target[1], target[0]])
                    if target not in G.edges():
                        print(f"Edge {target} not in graph. Skipping...")
                        continue
    
            G.remove_edge(*target) if isinstance(target, tuple) else G.remove_node(target)
            
            composite, robustness, reach, connectivity = GCI(G, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init)
            if exclude_benchmark:
                results_df.loc[i] = [i, set(target), composite, robustness, reach, connectivity, 'max_flow']
                continue

            for benchmark in benchmark_graphs:
                benchmark.remove_edge(*target)
                b_composite, b_robustness, b_reach, b_connectivity = GCI(benchmark, lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, dia_Gb_init, comp_centrs_Gb_init)
                
                b_connectedness_lst.append(b_composite)
                b_robustness_lst.append(b_robustness)
                b_reach_lst.append(b_reach)
                b_connectivity_lst.append(b_connectivity)

            results_df.loc[i] = [i, target, composite, robustness, reach, connectivity, sum(b_connectedness_lst) / len(b_connectedness_lst), sum(b_robustness_lst) / len(b_robustness_lst), sum(b_reach_lst) / len(b_reach_lst), sum(b_connectivity_lst) / len(b_connectivity_lst)]

    return results_df, results_best_worst_df

def GCI(G, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centr_G_init):
    largest_component_size, num_weakly_connected, num_strongly_connected, diameter, node_composite_centrality = get_connectedness_metrics_of(G)

    """
    RESILIENCE 
    Measures the network's resilience against random failures or targeted attacks. 
    It's approximated by the relative sizes of the largest connected component, weakly connected components, and strongly connected components.

    # CHANGE FROM PROJECT THESIS: instead of multiplying the indices, we take the geometric mean of the indices
    """
    if num_weakly_connected == 0:
        num_weakly_connected = 1
    if num_strongly_connected == 0:
        num_strongly_connected = 1
        
    GCI_robustness = ((largest_component_size / lcs_G_init) * (nwc_G_init / num_weakly_connected) * ( nsc_G_init / num_strongly_connected))**(1/3)

    """
    REACH 
    Quantifies the efficiency of information flow or reachability across the network. 
    It's reflected in the ratio of current average shortest path length and network diameter in comparison to their initial values.
    """
    GCI_reach = (diameter / dia_G_init)

    """
    CONNECTIVITY 
    Represents the centrality or average connectivity of the network. 
    It's estimated by comparing the current median node degree to the initial median node degree.
    """
    GCI_connectivity = (node_composite_centrality / comp_centr_G_init)


    """
    Calculate the composite connectedness index using the above calculated indices
    """
    GCI = (GCI_robustness * GCI_reach * GCI_connectivity)**(1/3)
    
    return GCI, GCI_robustness, GCI_reach, GCI_connectivity

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
    

#------------------------------------------------------------FROM HERE ONWARDS ARE HELPER FUNCTIONS------------------------------------------------------------

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

    node_composite_centrality = np.average(np.array(list(dict(CCI(G)).values())))
    return largest_component_size, num_weakly_connected, num_strongly_connected, diameter, node_composite_centrality


def get_banchmark(G, model):
    """
    Generates and returns a benchmark graph based on the given model.
    """
    if model == 'ER':
        
        n = len(G.nodes)
        p = len(G.edges) / (n * (n - 1))
        
        return nx.erdos_renyi_graph(n, p, directed=True, seed=SEED)

    if model == 'BA':
        n  = G.number_of_nodes()
        e = G.number_of_edges()
        m  = int(e / n)
        return nx.barabasi_albert_graph(n, m)
    
#------------------------------------------------------------FROM HERE ONWARDS ARE FUNCTIONS FOR PLOTTING------------------------------------------------------------
    
def plot_connectedness_fourway(results_dfs, titles, title_prefix="", include_benchmark=False):
    colors_set1 = ['peachpuff', 'powderblue']  
    colors_set2 = ['orange', 'dodgerblue']  
    
    fig, axs = plt.subplots(2, 2, figsize=SUB_PLOTS_FIGSIZE)
    axs = axs.flatten()

    metric_columns_list = [['composite', 'composite'], ['robustness', 'robustness'], ['reach', 'reach'], ['connectivity', 'connectivity']]
    benchmark_columns_list = [['composite_b', 'composite_b'], ['robustness_b', 'robustness_b'], ['reach_b', 'reach_b'], ['connectivity_b', 'connectivity_b']]
    metric_labels_list = [['random', 'greedy'], ['random', 'greedy'], ['random', 'greedy'], ['random', 'greedy']]
    benchmark_labels_list = [['ER grid random', 'ER grid greedy'], ['ER grid random', 'ER grid greedy'], ['ER grid random', 'ER grid greedy'], ['ER grid random', 'ER grid greedy']]
    ylabel = ['connectedness', 'robustness', 'reach', 'connectivity']

    # Find the minimum x-axis limit where the 'composite' metric reaches 0
    min_x_limit = float('inf')
    for results_df in results_dfs:
        min_x_limit = min(min_x_limit, results_df.loc[results_df['composite'] <= 0, 'iteration'].min())
    
    for i, ax in enumerate(axs):
        metric_columns = metric_columns_list[i]
        metric_labels = metric_labels_list[i]
        if include_benchmark:
            benchmark_columns = benchmark_columns_list[i]
            benchmark_labels = benchmark_labels_list[i]
        
        for j, results_df in enumerate(results_dfs):
            if j == 0:
                ax.plot(results_df['iteration'], results_df[metric_columns[j]], label=metric_labels[j], color=colors_set1[0])
                if include_benchmark:
                    ax.plot(results_df['iteration'], results_df[benchmark_columns[j]], label=benchmark_labels[j], color=colors_set1[1])
            else:
                ax.plot(results_df['iteration'], results_df[metric_columns[j]], label=metric_labels[j], color=colors_set2[0])
                if include_benchmark:
                    ax.plot(results_df['iteration'], results_df[benchmark_columns[j]], label=benchmark_labels[j], color=colors_set2[1])
        
        ax.set_xlabel('k iterations')
        ax.set_ylabel(ylabel[i])
        ax.set_title(titles[i])  

        ax.set_xlim(left=0, right=min_x_limit)
        
        if i == 0:
            ax.legend()
        
    plt.suptitle(title_prefix, x=SUP_TITLE_X, ha=SUP_TITLE_HA, fontsize=SUP_TITLE_FONTSIZE)
    plt.tight_layout()
    plt.show()
    return fig

def results_summary(df_, metric='', abs_or_pct='abs'):

    df = df_.copy()

    heuristic = 'entity criticality index'
    if 'max_flow_value' in df.columns:
        metric = 'max_flow_value'
        heuristic = df.iloc[1]['heuristic']

    zero_metric_iteration = df[df[metric] == 0].index.min()

    
    # Calculate differences between consecutive rows for the metric
    df['diff'] = round(df[metric].diff() * (-1), 2) 
    df['pct_change'] = round((df['diff'] / df[metric].shift(1))*-100, 1)
    df['it'] = df.index 

    if heuristic is not None:
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