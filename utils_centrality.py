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

def n_minus_k(G_, heuristic, remove, n_benchmarks=20, k_removals=250, exclude_benchmark=True, best_worst_case=False, er_best_worst=False, print_output=False, SEED=42):

    G = G_.copy()

    results_df = pd.DataFrame(columns=['iteration', 'removed_entity', 'composite', 'robustness', 'reach', 'connectivity', 'composite_b', 'robustness_b', 'reach_b', 'connectivity_b'])
    results_best_worst_df = pd.DataFrame(columns=['iteration', 'best_entity', 'composite_best', 'worst_entity', 'composite_worst'])

    if exclude_benchmark:
        results_df = pd.DataFrame(columns=['iteration', 'removed_entity', 'composite', 'robustness', 'reach', 'connectivity'])

    def assess_grid_connectedness_init(G):
        weakly_connected = list(nx.weakly_connected_components(G))
        largest_component = max(weakly_connected, key=len)
        largest_component_graph = G.subgraph(largest_component)

        strongly_connected = list(nx.strongly_connected_components(G))
        largest_strongly_connected = max(strongly_connected, key=len)
        largest_strongly_connected_graph = G.subgraph(largest_strongly_connected)

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
            diameter = G.number_of_edges() / nx.diameter(G.to_undirected())
        average_shortest_path_length = 0

        node_composite_centrality = np.average(np.array(list(dict(CCI(G)).values())))
        return len(largest_component), len(weakly_connected), len(strongly_connected), average_shortest_path_length, diameter, node_composite_centrality
    
    def find_best_and_worst(find_best, current_graph, G_init, lcs, nwc, nsc, aspl, dia, comp_centrs, remove_node=True):
        best_score, worst_score = float('-inf'), float('inf')
        entity = None
        
        for entity in current_graph.nodes() if remove_node else current_graph.edges():
            modified_graph = current_graph.copy()
            if remove_node:
                modified_graph.remove_node(entity)
            else:
                modified_graph.remove_edge(*entity)
            
            composite, _, _, _ = GCI(modified_graph, G_init, lcs, nwc, nsc, aspl, dia, comp_centrs)
            
            if np.isnan(composite):
                continue
            
            if find_best:
                if composite > best_score:
                    best_score = composite
                    entity_ = entity
            else:
                if composite < worst_score:
                    worst_score = composite
                    entity_ = entity
        
        if find_best:
            return best_score, entity_
        return worst_score, entity_


    benchmark_graphs = [get_banchmark(G.to_undirected(), model='ER') for _ in range(n_benchmarks)]
    r_graphs = [G.copy() for _ in range(n_benchmarks)]

    lcs_G_init, nwc_G_init, nsc_G_init, aspl_G_init, dia_G_init, comp_centrs_G_init = assess_grid_connectedness_init(G)
    lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, aspl_Gb_init, dia_Gb_init, comp_centrs_Gb_init = assess_grid_connectedness_init(benchmark_graphs[0])

    G_init = G.copy()
    Gb_init = benchmark_graphs[0].copy()

    r_best, r_worst = G.copy(), G.copy()
    lcs_BW, nwc_BW, nsc_BW, aspl_BW, dia_BW, comp_centrs_BW = lcs_G_init, nwc_G_init, nsc_G_init, aspl_G_init, dia_G_init, comp_centrs_G_init
    worst_case_terminated = False

    if best_worst_case and er_best_worst:
        r_best = Gb_init.copy()
        r_worst = Gb_init.copy()
        lcs_BW, nwc_BW, nsc_BW, aspl_BW, dia_BW, comp_centrs_BW = lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, aspl_Gb_init, dia_Gb_init, comp_centrs_Gb_init

    for i in tqdm(range(0, k_removals + 1), desc='N-k iterations'):  # Loop through k_removals + 1 iterations
        b_connectedness_lst, b_robustness_lst, b_reach_lst, b_connectivity_lst = [], [], [], []
        r_connectedness_lst, r_robustness_lst, r_reach_lst, r_connectivity_lst = [], [], [], []

        # For the first iteration, calculate and store initial connectedness indices
        if i == 0:  
            composite, robustness, reach, connectivity = GCI(G, G_init, lcs_G_init, nwc_G_init, nsc_G_init, aspl_G_init, dia_G_init, comp_centrs_G_init)

            if exclude_benchmark:
                results_df.loc[i] = [i, None, composite, robustness, reach, connectivity]
                continue

            composite_b, robustness_b, reach_b, connectivity_b = GCI(benchmark_graphs[0], Gb_init, lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, aspl_Gb_init, dia_Gb_init, comp_centrs_Gb_init)
            
            results_df.loc[i] = [i, None, composite, robustness, reach, connectivity, composite_b, robustness_b, reach_b, connectivity_b]
            results_best_worst_df.loc[i] = [i, None, composite, None, composite]
            continue  

        # Rest of iterations (i.e., not i == 0)
        if heuristic == 'random':

            if best_worst_case:
                
                if remove == 'node':
                    is_node = True
                else:
                    is_node = False

                best_score, best_entity = find_best_and_worst(True, r_best, G_init, lcs_BW, nwc_BW, nsc_BW, aspl_BW, dia_BW, comp_centrs_BW, remove_node=is_node)

                if is_node:
                    r_best.remove_node(best_entity)
                else:
                    r_best.remove_edge(*best_entity)

                if worst_case_terminated:
                    worst_score = 0
                else:
                    worst_score, worst_entity = find_best_and_worst(False, r_worst, G_init, lcs_BW, nwc_BW, nsc_BW, aspl_BW, dia_BW, comp_centrs_BW, remove_node=is_node)

                    if worst_entity is not None:
                        if is_node:
                            r_worst.remove_node(worst_entity)
                        else:
                            r_worst.remove_edge(*worst_entity)

                results_best_worst_df.loc[i] = [i, best_entity, best_score, worst_entity, worst_score]


            else:
                for real_copy in r_graphs:  # Iterate through the instantiated copies
                    target = random.choice(list(real_copy.nodes() if remove == 'node' else real_copy.edges()))

                    if print_output:
                        if remove == 'node':
                            print('Node removed at random iteration '+str(i)+': '+str(target))
                        else:
                            print('Edge removed at random iteration '+str(i)+': '+str(target))

                    real_copy.remove_node(target) if remove == 'node' else real_copy.remove_edge(*target)
                    r_composite, r_robustness, r_reach, r_connectivity = GCI(real_copy, G_init, lcs_G_init, nwc_G_init, nsc_G_init, aspl_G_init, dia_G_init, comp_centrs_G_init)
                    
                    # Log indices for each modified graph
                    r_connectedness_lst.append(r_composite)
                    r_robustness_lst.append(r_robustness)
                    r_reach_lst.append(r_reach)
                    r_connectivity_lst.append(r_connectivity)
                
                if exclude_benchmark:
                    results_df.loc[i] = [i, None,
                                        sum(r_connectedness_lst) / len(r_connectedness_lst), sum(r_robustness_lst) / len(r_robustness_lst), sum(r_reach_lst) / len(r_reach_lst), sum(r_connectivity_lst) / len(r_connectivity_lst)]
                    continue
                
                
                for benchmark in benchmark_graphs:
                    target_b = random.choice(list(benchmark.nodes() if remove == 'node' else benchmark.edges()))
                    benchmark.remove_node(target_b) if remove == 'node' else benchmark.remove_edge(*target_b)
                    b_composite, b_robustness, b_reach, b_connectivity = GCI(benchmark, Gb_init, lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, aspl_Gb_init, dia_Gb_init, comp_centrs_Gb_init)
                    
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

            if print_output:
                if remove == 'node':
                    print('Node removed at greedy iteration '+str(i)+': '+str(target))
                else:
                    print('Edge removed at greedy iteration '+str(i)+': '+str(target))

            G.remove_node(target) if remove == 'node' else G.remove_edge(*target)
            target = target if remove == 'node' else set(target)                
            composite, robustness, reach, connectivity = GCI(G, G_init, lcs_G_init, nwc_G_init, nsc_G_init, aspl_G_init, dia_G_init, comp_centrs_G_init)

            if exclude_benchmark:
                results_df.loc[i] = [i, target, composite, robustness, reach, connectivity]
                continue

            for benchmark in benchmark_graphs:
                benchmark_target = func(benchmark)
                benchmark.remove_node(benchmark_target) if remove == 'node' else benchmark.remove_edge(*benchmark_target)
                b_composite, b_robustness, b_reach, b_connectivity = GCI(benchmark, Gb_init, lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, aspl_Gb_init, dia_Gb_init, comp_centrs_Gb_init)
                
                b_connectedness_lst.append(b_composite)
                b_robustness_lst.append(b_robustness)
                b_reach_lst.append(b_reach)
                b_connectivity_lst.append(b_connectivity)

            results_df.loc[i] = [i, target, composite, robustness, reach, connectivity, sum(b_connectedness_lst) / len(b_connectedness_lst), sum(b_robustness_lst) / len(b_robustness_lst), sum(b_reach_lst) / len(b_reach_lst), sum(b_connectivity_lst) / len(b_connectivity_lst)]

    return results_df, results_best_worst_df

def GCI(G, G_init, lcs_G_init, nwc_G_init, nsc_G_init, aspl_G_init, dia_G_init, comp_centr_G_init):
    largest_component_size, num_weakly_connected, num_strongly_connected, average_shortest_path_length, diameter, node_composite_centrality = get_connectedness_metrics_of(G)

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

    # CHANGE FROM PROJECT THESIS: reach calculated correctly in terms of multiple components, using solely diameter
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

    # CHANGE FROM PROJECT THESIS: instead of multiplying the indices, we take the geometric mean of the indices

    """
    GCI = (GCI_robustness * GCI_reach * GCI_connectivity)**(1/3)
    
    return GCI, GCI_robustness, GCI_reach, GCI_connectivity

def CCI(G):
    """
    Returns a dictionary of composite centrality values for each node in the given graph G
    """
    node_degree = nx.degree_centrality(G)
    betweenness = nx.betweenness_centrality(create_digraph_of(G), normalized=True)
    closeness = nx.closeness_centrality(G)
    epsilon = 1e-6
    composite_centralities = {
        country: node_degree.get(country, epsilon) * betweenness.get(country, epsilon) * closeness.get(country, epsilon) for country in G.nodes()
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

def create_digraph_of(G):
    """
    Creates and returns a weighted nx.DiGraph from the given nx.MultiDiGraph G, where each edge has an aggregated flow attribute.
    """
    weighted_graph = nx.DiGraph()
    for u, v, data in G.edges(data=True):
        if not weighted_graph.has_edge(u, v):
            weighted_graph.add_edge(u, v, flow=0)
            weighted_graph[u][v]['flow'] += data.get('flow', 0)
    return weighted_graph

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

    strongly_connected = list(nx.strongly_connected_components(G))
    largest_strongly_connected = max(strongly_connected, key=len, default=[])
    largest_strongly_connected_graph = G.subgraph(largest_strongly_connected)

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
        diameter = G.number_of_edges() / nx.diameter(G.to_undirected())

    average_shortest_path_length = nx.average_shortest_path_length(largest_strongly_connected_graph.to_undirected())

    node_composite_centrality = np.average(np.array(list(dict(CCI(G)).values())))
    return largest_component_size, num_weakly_connected, num_strongly_connected, average_shortest_path_length, diameter, node_composite_centrality


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
    metric_labels_list = [['Real grid random', 'Real grid greedy'], ['Real grid random', 'Real grid greedy'], ['Real grid random', 'Real grid greedy'], ['Real grid random', 'Real grid greedy']]
    benchmark_labels_list = [['ER grid random', 'ER grid greedy'], ['ER grid random', 'ER grid greedy'], ['ER grid random', 'ER grid greedy'], ['ER grid random', 'ER grid greedy']]
    ylabel = ['connectedness', 'robustness', 'reach', 'connectivity']
    
    for i, ax in enumerate(axs):
        metric_columns = metric_columns_list[i]
        metric_labels = metric_labels_list[i]
        if include_benchmark:
            benchmark_columns = benchmark_columns_list[i]
            benchmark_labels = benchmark_labels_list[i]
        
        for j, results_df in enumerate(results_dfs):
            if j == 0:
                ax.plot(results_df['iteration'], results_df[metric_columns[j]], marker='o', label=metric_labels[j], color=colors_set1[0])
                if include_benchmark:
                    ax.plot(results_df['iteration'], results_df[benchmark_columns[j]], marker='o', label=benchmark_labels[j], color=colors_set1[1])
            else:
                ax.plot(results_df['iteration'], results_df[metric_columns[j]], marker='o', label=metric_labels[j], color=colors_set2[0])
                if include_benchmark:
                    ax.plot(results_df['iteration'], results_df[benchmark_columns[j]], marker='o', label=benchmark_labels[j], color=colors_set2[1])
        
        ax.set_xlabel('k iterations')
        ax.set_ylabel(ylabel[i])
        ax.grid(True)
        ax.set_title(titles[i])  
        
        if i == 0:
            ax.legend()
        
    plt.suptitle(title_prefix, x=SUP_TITLE_X, ha=SUP_TITLE_HA, fontsize=SUP_TITLE_FONTSIZE)
    plt.tight_layout()
    plt.show()

def compare_scigrid_entsog(data, metric=None):
   
    scigrid_columns = ['iteration_scigrid', 'composite_scigrid', 'robustness_scigrid', 'reach_scigrid', 'connectivity_scigrid']
    entsog_columns = ['iteration_entsog', 'composite_entsog', 'robustness_entsog', 'reach_entsog', 'connectivity_entsog']

    df = data.copy()

    df['iteration_scigrid_scaled'] = (df['iteration_scigrid'] - df['iteration_scigrid'].min()) / (df['iteration_scigrid'].max() - df['iteration_scigrid'].min())
    df['iteration_entsog_scaled'] = (df['iteration_entsog'] - df['iteration_entsog'].min()) / (df['iteration_entsog'].max() - df['iteration_entsog'].min())

    # Plotting
    plt.figure(figsize=(10, 6))

    if metric == None:
        for i, col in enumerate(scigrid_columns[1:]):
            plt.plot(df['iteration_scigrid_scaled'], df[col], label=col, linestyle='-', color=f'C{i}', linewidth=2)

        # Plotting entsog
        for i, col in enumerate(entsog_columns[1:]):
            plt.plot(df['iteration_entsog_scaled'], df[col], label=col, linestyle='--', color=f'C{i}', linewidth=2)

    else:
        plt.plot(df['iteration_scigrid_scaled'], df[metric+'_scigrid'], label=metric+'_scigrid', linestyle='-', color='C0', linewidth=2)
        plt.plot(df['iteration_entsog_scaled'], df[metric+'_entsog'], label=metric+'_entsog', linestyle='--', color='C0', linewidth=2)

    plt.xlabel('Scaled Iteration')
    plt.ylabel('Metrics')
    plt.suptitle('N-k project thesis algorithm, Scigrid and Entsog comparison', x=.375)
    plt.title(data.name, loc='left')
    plt.legend()
    plt.grid(True)
    plt.show()

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