from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import random
SEED=42

def create_graphs_from_dataset(df):
    """
    Creates a list of nx.MultiDiGraphs from the given dataset.
    Each nx.MultiDiGraph represents a month-year combination.
    """
    graphs = []
    mm_yyyy = df.iloc[:, df.columns.get_loc('Jan-10'):]

    for index, row in df.iterrows():
        seen_nonzero = False
        for col in mm_yyyy.columns:
            if row[col] == 0:
                if not seen_nonzero:
                    df.at[index, col] = -1
            else:
                seen_nonzero = True
                if pd.isna(row[col]):
                    df.at[index, col] = 0

    for col in mm_yyyy.columns:
        G = nx.MultiDiGraph()
        non_zero_rows = df[(df[col] != -1) & (~df[col].isna())]

        for _, row in non_zero_rows.iterrows():
            borderpoint, exit_country, entry_country, max_flow, flow = row['Borderpoint'], row['Exit'], row['Entry'], row['MAXFLOW (Mm3/h)'], row[col]

            G.add_node(exit_country)
            G.add_node(entry_country)

            G.add_edge(exit_country, entry_country, borderpoint=borderpoint, flow=flow, max_flow=float(max_flow))

        graphs.append(G)
    return graphs


#------------------------------------------------------------FROM HERE ONWARDS IS CODE FROM PROJECT THESIS------------------------------------------------------------

def n_minus_k(G, n_benchmarks, k_removals, heuristic, remove, best_worst_case=False, er_best_worst=False):

    results_df = pd.DataFrame(columns=['iteration', 'composite', 'robustness', 'reach', 'connectivity', 'composite_b', 'robustness_b', 'reach_b', 'connectivity_b'])
    results_best_worst_df = pd.DataFrame(columns=['iteration', 'best_entity', 'composite_best', 'worst_entity', 'composite_worst'])

    def assess_grid_connectedness_init(G):
        weakly_connected = list(nx.weakly_connected_components(G))
        largest_component = max(weakly_connected, key=len)
        largest_component_graph = G.subgraph(largest_component)

        return len(largest_component), len(weakly_connected), len(list(nx.strongly_connected_components(G))), nx.average_shortest_path_length(largest_component_graph), nx.diameter(largest_component_graph.to_undirected()), np.average(np.array(list(list(dict(CCI(G)).values()))))
    
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


    benchmark_graphs = [ER_benchmark(G) for _ in range(n_benchmarks)]
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

    for i in range(k_removals + 1):  # Loop through k_removals + 1 iterations
        b_connectedness_lst, b_robustness_lst, b_reach_lst, b_connectivity_lst = [], [], [], []
        r_connectedness_lst, r_robustness_lst, r_reach_lst, r_connectivity_lst = [], [], [], []

        # For the first iteration, calculate and store initial connectedness indices
        if i == 0:  
            composite, robustness, reach, connectivity = GCI(G, G_init, lcs_G_init, nwc_G_init, nsc_G_init, aspl_G_init, dia_G_init, comp_centrs_G_init)
            composite_b, robustness_b, reach_b, connectivity_b = GCI(benchmark_graphs[0], Gb_init, lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, aspl_Gb_init, dia_Gb_init, comp_centrs_Gb_init)
            
            results_df.loc[i] = [i, composite, robustness, reach, connectivity, composite_b, robustness_b, reach_b, connectivity_b]
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
                
                
                for benchmark in benchmark_graphs:
                    target_b = random.choice(list(benchmark.nodes() if remove == 'node' else benchmark.edges()))
                    benchmark.remove_node(target_b) if remove == 'node' else benchmark.remove_edge(*target_b)
                    b_composite, b_robustness, b_reach, b_connectivity = GCI(benchmark, Gb_init, lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, aspl_Gb_init, dia_Gb_init, comp_centrs_Gb_init)
                    
                    b_connectedness_lst.append(b_composite)
                    b_robustness_lst.append(b_robustness)
                    b_reach_lst.append(b_reach)
                    b_connectivity_lst.append(b_connectivity)

                results_df.loc[i] = [i, 
                                    sum(r_connectedness_lst) / len(r_connectedness_lst), sum(r_robustness_lst) / len(r_robustness_lst), sum(r_reach_lst) / len(r_reach_lst), sum(r_connectivity_lst) / len(r_connectivity_lst), 
                                    sum(b_connectedness_lst) / len(b_connectedness_lst), sum(b_robustness_lst) / len(b_robustness_lst), sum(b_reach_lst) / len(b_reach_lst), sum(b_connectivity_lst) / len(b_connectivity_lst)]



        elif heuristic == 'greedy':
            func = CCI_v if remove == 'node' else CCI_e
            target = func(G)

            if remove == 'node':
                print('Node removed at greedy iteration '+str(i)+': '+str(target))
            else:
                print('Edge removed at greedy iteration '+str(i)+': '+str(target))

            G.remove_node(target) if remove == 'node' else G.remove_edge(*target)
            
            composite, robustness, reach, connectivity = GCI(G, G_init, lcs_G_init, nwc_G_init, nsc_G_init, aspl_G_init, dia_G_init, comp_centrs_G_init)

            
            for benchmark in benchmark_graphs:
                benchmark_target = func(benchmark)
                benchmark.remove_node(benchmark_target) if remove == 'node' else benchmark.remove_edge(*benchmark_target)
                b_composite, b_robustness, b_reach, b_connectivity = GCI(benchmark, Gb_init, lcs_Gb_init, nwc_Gb_init, nsc_Gb_init, aspl_Gb_init, dia_Gb_init, comp_centrs_Gb_init)
                
                b_connectedness_lst.append(b_composite)
                b_robustness_lst.append(b_robustness)
                b_reach_lst.append(b_reach)
                b_connectivity_lst.append(b_connectivity)

            results_df.loc[i] = [i, composite, robustness, reach, connectivity, sum(b_connectedness_lst) / len(b_connectedness_lst), sum(b_robustness_lst) / len(b_robustness_lst), sum(b_reach_lst) / len(b_reach_lst), sum(b_connectivity_lst) / len(b_connectivity_lst)]

    return results_df, results_best_worst_df

def GCI(G, G_init, lcs_G_init, nwc_G_init, nsc_G_init, aspl_G_init, dia_G_init, comp_centr_G_init):
    largest_component_size, num_weakly_connected, num_strongly_connected, average_shortest_path_length, diameter, node_composite_centrality = get_connectedness_metrics_of(G)

    """
    RESILIENCE 
    Measures the network's resilience against random failures or targeted attacks. 
    It's approximated by the relative sizes of the largest connected component, weakly connected components, and strongly connected components.
    """
    GCI_robustness = (largest_component_size / lcs_G_init) * (nwc_G_init / num_weakly_connected) * ( nsc_G_init / num_strongly_connected)

    """
    REACH 
    Quantifies the efficiency of information flow or reachability across the network. 
    It's reflected in the ratio of current average shortest path length and network diameter in comparison to their initial values.
    """
    GCI_reach = (average_shortest_path_length / aspl_G_init) * ((G.number_of_edges() / diameter) / (G_init.number_of_edges() / dia_G_init))


    """
    CONNECTIVITY 
    Represents the centrality or average connectivity of the network. 
    It's estimated by comparing the current median node degree to the initial median node degree.
    """
    GCI_connectivity = (node_composite_centrality / comp_centr_G_init)


    """
    Calculate the composite connectedness index using the above calculated indices
    """
    GCI = GCI_robustness * GCI_reach * GCI_connectivity
    
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

def CCI_e(G):
    """
    Returns the edge with the highest composite centrality value in the given graph G
    """
    centrality = CCI(G)
    edge_centrality = {
        edge: (centrality[edge[0]] + centrality[edge[1]]) / 2.0
        for edge in G.edges()
    }
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
    - Average shortest path length
    - Diameter
    - Node composite centrality
    """
    weakly_connected = list(nx.weakly_connected_components(G))
    largest_component = max(weakly_connected, key=len)
    largest_component_size = len(largest_component)
    num_weakly_connected = len(weakly_connected)
    num_strongly_connected = len(list(nx.strongly_connected_components(G)))

    if largest_component_size > 1:
        largest_component_graph = G.subgraph(largest_component)
        average_shortest_path_length = nx.average_shortest_path_length(largest_component_graph)
        diameter = nx.diameter(largest_component_graph.to_undirected())
        node_composite_centrality = np.average(np.array(list(dict(CCI(G)).values())))
    else:
        average_shortest_path_length = diameter = node_composite_centrality = 0

    return largest_component_size, num_weakly_connected, num_strongly_connected, average_shortest_path_length, diameter, node_composite_centrality


def ER_benchmark(G):
    """
    Generates and returns a random directed graph using the Erdős-Rényi model.
    """
    n = len(G.nodes)
    p = len(G.edges) / (n * (n - 1))
    
    return nx.erdos_renyi_graph(n, p, directed=True, seed=SEED)

def plot_connectedness_fourway(results_dfs, titles):
    colors_set1 = ['peachpuff', 'powderblue']  # Custom colors for the first set of data
    colors_set2 = ['orange', 'dodgerblue']  # Custom colors for the second set of data
    
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    metric_columns_list = [['composite', 'composite'], ['robustness', 'robustness'], ['reach', 'reach'], ['connectivity', 'connectivity']]
    benchmark_columns_list = [['composite_b', 'composite_b'], ['robustness_b', 'robustness_b'], ['reach_b', 'reach_b'], ['connectivity_b', 'connectivity_b']]
    metric_labels_list = [['Real grid random', 'Real grid greedy'], ['Real grid random', 'Real grid greedy'], ['Real grid random', 'Real grid greedy'], ['Real grid random', 'Real grid greedy']]
    benchmark_labels_list = [['ER grid random', 'ER grid greedy'], ['ER grid random', 'ER grid greedy'], ['ER grid random', 'ER grid greedy'], ['ER grid random', 'ER grid greedy']]
    ylabel = ['connectedness', 'robustness', 'reach', 'connectivity']
    
    for i, ax in enumerate(axs):
        metric_columns = metric_columns_list[i]
        benchmark_columns = benchmark_columns_list[i]
        metric_labels = metric_labels_list[i]
        benchmark_labels = benchmark_labels_list[i]
        
        for j, results_df in enumerate(results_dfs):
            if j == 0:
                ax.plot(results_df['iteration'], results_df[metric_columns[j]], marker='o', label=metric_labels[j], color=colors_set1[0])
                ax.plot(results_df['iteration'], results_df[benchmark_columns[j]], marker='o', label=benchmark_labels[j], color=colors_set1[1])
            else:
                ax.plot(results_df['iteration'], results_df[metric_columns[j]], marker='o', label=metric_labels[j], color=colors_set2[0])
                ax.plot(results_df['iteration'], results_df[benchmark_columns[j]], marker='o', label=benchmark_labels[j], color=colors_set2[1])
        
        ax.set_xlabel('k iterations', fontsize=14)
        ax.set_ylabel(ylabel[i], fontsize=14)
        ax.grid(True)
        ax.set_title(titles[i], fontsize=16)  
        
        if i == 0:
            ax.legend()
        
    plt.tight_layout()
    plt.show()



    


