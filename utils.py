from matplotlib import pyplot as plt
import pandas as pd
import geopandas as gpd
import geojson as gj
import json
import numpy as np
import networkx as nx
import random
import matplotlib.image as mpimg
SEED=42
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}

###------------------------------------------------------------FROM HERE ARE FUNCTIONS TO LOAD DATA------------------------------------------------------------###

def get_IGGIN_pipeline_data():
    """
    Loads the IGGIN pipeline dataset and returns it as a pandas DataFrame.
    """
    def process_column(df, column, prefix=''):
        df[column] = df[column].apply(lambda x: json.loads(x.replace("'", '"')))
        normalized_df = pd.json_normalize(df[column])
        normalized_df.columns = [f'{prefix}{col}' for col in normalized_df.columns]
        df = df.drop(columns=[column]).join(normalized_df)
        return df

    pipelines_gdf = gpd.read_file('Scigrid_data/IGGIN_PipeSegments.csv')

    pipelines_gdf = process_column(pipelines_gdf, 'param')
    pipelines_gdf = process_column(pipelines_gdf, 'uncertainty', 'uncertainty_')
    pipelines_gdf = process_column(pipelines_gdf, 'method', 'method_')

    columns_to_split = ['country_code', 'node_id']
    for column in columns_to_split:
        pipelines_gdf[column] = pipelines_gdf[column].apply(lambda x: x.replace('[', '').replace(']', '').replace("'", '').split(', '))
        split_df = pd.DataFrame(pipelines_gdf[column].to_list(), columns=[f'{column}_1', f'{column}_2'])
        pipelines_gdf = pipelines_gdf.drop(columns=[column]).join(split_df)

    return pipelines_gdf



###------------------------------------------------------------FROM HERE ARE ALGORITHMS AND GETTERS------------------------------------------------------------###

def max_flow(graph, sources, sinks, flow_func=nx.algorithms.flow.dinitz, capacity='capacity', show_plot=True):

    """ Compute the maximum flow from sources to sinks in a graph and visualize the flow network.
    Args:
        graph: NetworkX graph
        sources: List of source nodes
        sinks: List of sink nodes
        flow_func: Function to use for computing the maximum flow
        capacity: Name of the edge attribute to use for flow capacity
        show_plot: Whether to display the plot or not
    Returns:
        flow_value: Value of the maximum flow
        flow_dict: Dictionary containing the flow value on each edge
        flow_edges: List of edges with non-zero flow    
    """

    """ def add_super_source_sink(graph, sources, sinks):
        super_source = "super_source"
        super_sink = "super_sink"

        # Create super source and add edges to all source nodes
        graph.add_node(super_source)
        for source in sources:
            graph.add_edge(super_source, source, capacity=float('inf'))

        # Create super sink and add edges from all sink nodes
        graph.add_node(super_sink)
        for sink in sinks:
            graph.add_edge(sink, super_sink, capacity=float('inf'))

        return super_source, super_sink
    
    super_source, super_sink = add_super_source_sink(graph, sources, sinks) """

    # Run max flow algorithm
    flow_value, flow_dict = nx.maximum_flow(graph, sources, sinks, capacity=capacity, flow_func=flow_func)

    # Extract edges with non-zero flow
    flow_edges = [(u, v) for u in flow_dict for v in flow_dict[u] if flow_dict[u][v] > 0]

    # Create a new graph with only relevant edges
    relevant_graph = graph.edge_subgraph(flow_edges)

    # Extract node positions if available
    pos = nx.get_node_attributes(graph, 'pos')

    # Define node and edge colors
    node_colors = ['red' if node in sources else 'yellow' if node in sinks else 'lightblue' for node in graph.nodes]
    edge_colors = ['red' if (u, v) in flow_edges else 'gray' for u, v in graph.edges]

    # Define edge labels
    labels = {(u, v): f'{flow_dict[u][v]:.1f}' for u, v in flow_edges}

    europe_map = mpimg.imread('Europe_blank_map.png')


    # Use plt.imshow to display the background map
    plt.figure(figsize=(15, 10))
    plt.imshow(europe_map, extent=[-20, 40, 35, 70], alpha=0.5)

    # Draw nodes and edges on top of the map
    nx.draw(graph, pos=pos, with_labels=False, node_size=70, node_color=node_colors, font_size=8,
            font_color="black", font_weight="bold", arrowsize=10, width=1, edge_color=edge_colors, alpha=0.7)
    
    # Draw relevant edges with a thicker red line
    nx.draw_networkx_edges(relevant_graph, pos=pos, edge_color='red', width=2)
    
    nx.draw_networkx_edge_labels(graph, pos=pos, edge_labels=labels, font_color='red', alpha=1, bbox=dict(alpha=0))


    title = f'Max Flow from {sources} to {sinks}: {flow_value:.1f}'
    plt.title(title, fontsize=20)

    if show_plot:
        plt.show()
    else:
        plt.close()

    return flow_value, flow_dict, flow_edges


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
        G = nx.MultiDiGraph(name=col)
        non_zero_rows = df[(df[col] != -1) & (~df[col].isna())]

        for _, row in non_zero_rows.iterrows():
            borderpoint, exit_country, entry_country, max_flow, flow = row['Borderpoint'], row['Exit'], row['Entry'], row['MAXFLOW (Mm3/h)'], row[col]

            if pd.isna(max_flow):
                max_flow = row.drop(['Borderpoint', 'Exit', 'Entry', 'MAXFLOW (Mm3/h)']).max()

            else:
                month, year = col.split('-')
                days_in_month = pd.Timestamp(year=int('20'+year), month=month_map[month], day=1).days_in_month
                max_flow = float(max_flow) * 24 * days_in_month

            G.add_node(exit_country)
            G.add_node(entry_country)

            G.add_edge(exit_country, entry_country, borderpoint=borderpoint, flow=flow, capacity=float(max_flow))

        # Add ENTSOG 2024 supply and demand data as node attributes
        G = add_node_attributes(G)

        # Fix edge capacities if flow exceeds max flow
        G = update_edge_capacities(G)

        graphs.append(G)
    return graphs

def get_edge_data(graph):
    edge_data = []

    for edge in graph.edges(data=True):
        source, target, edge_attributes = edge
        borderpoint = edge_attributes.get('borderpoint', None)
        max_flow = edge_attributes.get('capacity', None)
        flow = edge_attributes.get('flow', None)

        edge_data.append({'Source': source, 'Target': target, 'Borderpoint': borderpoint, 'Max Flow': max_flow, 'Flow': flow})

    return pd.DataFrame(edge_data)

def update_edge_capacities(graph):
    for u, v, data in graph.edges(data=True):
        flow = data['flow']
        max_flow = data['capacity']
        
        if flow > max_flow:
            data['capacity'] = flow
    return graph

def add_node_attributes(G):
    node_data = pd.read_csv('./Data/ENTSOG_2024_Supply_Demand.csv')

    for node in G.nodes():
        if node in node_data['Country'].values:
            node_attributes = node_data[node_data['Country'] == node].iloc[0]
            attributes = {
                'Total Demand': node_attributes['Total Demand (GWh)'],
                'Summer Demand': node_attributes['Summer Demand (GWh)'],
                'Winter Demand': node_attributes['Winter Demand (GWh)'],
                'Max Production': node_attributes['Max Production (GWh/d)'],
                'Storage Deliverability': node_attributes['Storage Capacities Deliverability (GWh/d)'],
                'Storage Injection': node_attributes['Storage Capacities Injection (GWh/d)'],
                'Storage WGV': node_attributes['Storage Capacities WGV (GWh)'],
                'LNG Send-out': node_attributes['LNG Capacities Send-out (GWh/d)'],
                'LNG Storage': node_attributes['LNG Capacities Storage (Mcm)'],
                'Power Generation': node_attributes['Power Generation (MWe)']
            }
            nx.set_node_attributes(G, {node: attributes})
    return G

def get_node_data(G):
    
    # Create a dataframe with the columns of the node attributes
    col_names = []
    for key, val in G.nodes.data():
        if len(val) > 0:
            for k, v in val.items():
                col_names.append(k)
            break

    # Populate the dataframe with the node attributes
    node_df = pd.DataFrame(columns=['Country'] + list(col_names))
    for node, attributes in G.nodes(data=True):
        node_info = {'Country': node}
        node_info.update(attributes)  

        node_df = pd.concat([node_df, pd.DataFrame([node_info])], ignore_index=True)    
    return node_df


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



    


