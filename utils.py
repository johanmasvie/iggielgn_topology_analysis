from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
import json
import numpy as np
import networkx as nx
import random
import matplotlib.image as mpimg


SEED=42
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
SUP_TITLE_X, SUP_TITLE_HA, SUP_TITLE_FONTSIZE = 0.12, 'center', 'x-large'
SUB_PLOTS_FIGSIZE = (12, 6)


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

def add_super_source_sink(G, sources, sinks):
    super_source = "super_source"
    super_sink = "super_sink"

    # Create super source and add edges to all source nodes
    G.add_node(super_source)
    for source in sources:
        G.add_edge(super_source, source, capacity=10000)

    # Create super sink and add edges from all sink nodes
    G.add_node(super_sink)
    for sink in sinks:
        G.add_edge(sink, super_sink, capacity=10000)

    return super_source, super_sink


def country_or_node_analysis(G, sources, sinks, all_to_all_flow):
    """
    Prepares the graph for analysis on country-level abstraction or 'sinks-to-sources' basis. 
    """
        
    # Prepare graph for analysis on country-level abstraction
    if any('is_country_node' in G.nodes[n] for n in G.nodes):
        # Remove the country node abstractions from the graph
        G.remove_nodes_from([n for n in G.nodes if G.nodes[n].get('is_country_node') and n not in sources and n not in sinks])

    # Prepare graph for analysis on 'sinks-to-sources' basis
    if all_to_all_flow:
        if any('is_country_node' in G.nodes[n] for n in G.nodes):
            # Remove the country node abstractions from the graph
            G.remove_nodes_from([n for n in G.nodes if G.nodes[n].get('is_country_node')])

        sources = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) > 0]
        sinks = [n for n in G.nodes() if G.out_degree(n) == 0 and G.in_degree(n) > 0]

    return G, sources, sinks

def add_country_node_abstraction(G):
    G_with_country_nodes = G.copy()

    country_positions = {}

    for node_id, node_data in G_with_country_nodes.nodes(data=True):
        country_code = node_data.get('country_code')
        if country_code is not None:
            country_code = str.strip(country_code.upper())
        
        if country_code not in country_positions:
            country_positions[country_code] = []
        country_positions[country_code].append(node_data['pos'])

    for country_code, positions in country_positions.items():
        average_position = np.mean(positions, axis=0)
        G_with_country_nodes.add_node(country_code, pos=average_position, is_country_node=True, country_code=country_code)

    for node_id, node_data in G_with_country_nodes.nodes(data=True):
        if 'country_node' in node_data:
            G_with_country_nodes.remove_node(node_id)

    # Get the list of country nodes
    country_nodes = [node_id for node_id, node_data in G_with_country_nodes.nodes(data=True) if node_data.get('is_country_node')]

    # Iterate over each node in the graph
    for node_id, node_data in G_with_country_nodes.nodes(data=True):
        # Skip country nodes
        if node_data.get('is_country_node'):
            continue
        
        # Get the country code of the node
        country_code = node_data.get('country_code')
        
        # Find the corresponding country super node
        country_super_node = next((cn for cn in country_nodes if G_with_country_nodes.nodes[cn]['country_code'] == country_code), None)
        
        if country_super_node:
            # Check if the node is a sink (only incoming edges)
            if G_with_country_nodes.in_degree(node_id) > 0 and G_with_country_nodes.out_degree(node_id) == 0:
                # Calculate the aggregate in-degree capacity of the child node
                aggregate_in_capacity = sum(G_with_country_nodes.edges[neighbor, node_id]['capacity'] for neighbor in G_with_country_nodes.predecessors(node_id))
                
                # Add an edge directed towards the country node from the node with the aggregate in-degree capacity
                G_with_country_nodes.add_edge(node_id, country_super_node, capacity = aggregate_in_capacity)
            
            
            # Check if the node is a source (only outgoing edges)
            if G_with_country_nodes.in_degree(node_id) == 0 and G_with_country_nodes.out_degree(node_id) > 0:
                # Calculate the aggregate out-degree capacity of the child node
                aggregate_out_capacity = sum(G_with_country_nodes.edges[node_id, neighbor]['capacity'] for neighbor in G_with_country_nodes.successors(node_id))
                
                # Add an edge directed towards the node from the country node with the aggregate out-degree capacity
                G_with_country_nodes.add_edge(country_super_node, node_id, capacity = aggregate_out_capacity) 
    
    return G_with_country_nodes


def max_flow_edge_count(G, prev_max_flow_vals, current_iteration, count_or_flow='count'):
    """
    Calculates the number of times each edge is part of the max flow path between any two nodes in the graph G. 

    Runtime approx. 63 minutes for N-k algorithm with complete dataset and 250 node removals, count_or_flow='count'.
    Runtime approx. -- minutes for N-k algorithm with complete dataset and 250 edge removals,  count_or_flow='count'.

    Runtime approx. 65 minutes for N-k algorithm with complete dataset and 250 node removals, count_or_flow='flow'.
    Runtime approx. 92 minutes for N-k algorithm with complete dataset and 250 edge removals, count_or_flow='flow'.
    """

    nodes = list(G.nodes)
    nodes.remove('super_source')
    nodes.remove('super_sink')
    n = len(nodes)

    edge_count = {(u, v): 0 for u, v in G.edges() if u not in ['super_source', 'super_sink'] and v not in ['super_sink', 'super_source']}
    
    for i in range(n):

        # The current node
        source = nodes[i]

        # Skip nodes with out-degree 0
        if G.out_degree(source) == 0:
            continue        

        for j in range(i+1, n):

            # Node the max flow is calculated to
            sink = nodes[j]
            
            if nx.has_path(G, source, sink):

                # If the previous max flow value is 0, the flow will be 0 now as well and the edge count will be 0
                if current_iteration > 1:
                    if (source, sink) in prev_max_flow_vals and prev_max_flow_vals[(source, sink)] == 0:
                        continue

                flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity='max_cap_M_m3_per_d')
                
                prev_max_flow_vals[(source, sink)] = flow_value
            
                for u, flows in flow_dict.items():
                    for v, flow in flows.items():
                        
                        if count_or_flow == 'flow':
                            if (u, v) in edge_count:
                                edge_count[(u, v)] += flow
                                continue

                        if flow > 0:
                            if (u, v) in edge_count:
                                if count_or_flow == 'count':
                                    edge_count[(u, v)] += 1
                                
    
    edge_count_raw = {k: v for k, v in edge_count.items()}
    
    edge_count_combined = {k: {'edge': k, 'max_flow_edge_count': v} for k, v in edge_count_raw.items()}
    
    df = pd.DataFrame.from_dict(edge_count_combined, orient='index').reset_index()

    if df.empty:
        return df, prev_max_flow_vals
    
    df.drop(columns=['level_0', 'level_1'], inplace=True)
    
    df = df.sort_values(by='max_flow_edge_count', ascending=False)
    return df, prev_max_flow_vals

def edge_cutset_count(G, observed_min_cutsets, current_iteration):
    """
    # TODO: something wonky about this heuristic?
    Calculates the number of times each edge is part of the minimum cutset between any two nodes in the graph G.

    Runtime approx. 104 minutes for N-k algorithm with complete dataset and 250 iterations.
    Runtime approx. 67 minutes for N-k algorithm with complete dataset and 250 node removals.

    """
    nodes = list(G.nodes)
    n = len(nodes)

    edge_cutset_count_ = {edge: 0 for edge in G.edges}

    for i in tqdm(range(n), desc='Calculating min cutset count'):

        source = nodes[i]

        if source == 'super_source' or source == 'super_sink':
            continue

        if G.out_degree(source) == 0:
            continue

        for j in range(i+1, n):

            sink = nodes[j]

            if sink == 'super_source' or sink == 'super_sink':
                continue

            if nx.has_path(G, source, sink):

                if source != sink:

                    if current_iteration > 1:
                        if observed_min_cutsets[(source, sink)] == 0:
                            continue

                    min_cutset = nx.minimum_edge_cut(G, source, sink)

                    observed_min_cutsets[(source, sink)] = len(min_cutset)

                    for edge in G.edges:
                        if edge in min_cutset:
                            edge_cutset_count_[edge] += 1

    data = [{
        'edge': edge,
        'min_cutset_count': value,
    } for edge, value in edge_cutset_count_.items()]

    df = pd.DataFrame(data)
    df = df.sort_values(by='min_cutset_count', ascending=False)

    return df, observed_min_cutsets


def weighted_flow_capacity_rate(G):
    """ 
    Calculates the Weighted Flow Capacity Robustness (WFCR) of the graph G.    
    Runtime 71m for N-k algorithm with complete dataset and 250 node removals.
    """

    nodes = list(G.nodes)
    n = len(nodes)

    edge_WFCR = {}
    tot_flow = 0

    for i in tqdm(range(n), desc='Calculating wfcr'):

        source = nodes[i]
        if source == 'super_source' or source == 'super_sink':
            continue

        for j in range(i + 1, n):

            sink = nodes[j]

            if sink == 'super_source' or sink == 'super_sink':
                    continue


            if nx.has_path(G, source, nodes[j]):
                
                flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity='max_cap_M_m3_per_d')
                tot_flow += flow_value
                
                for u, flows in flow_dict.items():
                    for v, flow in flows.items():
                        if flow > 0:
                            if (u, v) in G.edges:
                                capacity = G.edges[(u, v)]['max_cap_M_m3_per_d']
                            
                            if capacity > 0:  
                                edge_WFCR[(u, v)] = edge_WFCR.get((u, v), 0) + (flow ** 2) / capacity
                            else:
                                pass

    if tot_flow == 0:
        return pd.DataFrame()
    
    data = [{
        'edge': k,
        'wfcr': v / (n * (n - 1) * tot_flow),
    } for k, v in edge_WFCR.items()]

    df = pd.DataFrame(data)
    df = df.sort_values(by='wfcr', ascending=False)

    return df

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

def plot_heuristic_comparison_biplot(df_list, title_prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=SUB_PLOTS_FIGSIZE)

    series_names = list(df_list[0].columns[:2])

    for ax, series_name in zip(axes, series_names):
        for df in df_list:
            heuristic = str(df.iloc[1]['heuristic'])
            remove = 'edge' if isinstance(df.iloc[1]['removed_entity'], tuple) else 'node'
            ax.plot(df.index, df[series_name], marker='o', label=f'{heuristic} {remove} removals')

        ax.set_xlabel('k iterations')
        ax.set_ylabel(series_name.replace('_', ' '))
        ax.set_title(f'{series_name.replace("_", " ").title()} vs k removals')

        ax.legend()

    plt.suptitle(title_prefix, x=SUP_TITLE_X, ha=SUP_TITLE_HA, fontsize=SUP_TITLE_FONTSIZE)
    plt.tight_layout()
    plt.show()

def plot_connectedness_fourway(results_dfs, titles, title_prefix=""):
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
        
        ax.set_xlabel('k iterations')
        ax.set_ylabel(ylabel[i])
        ax.grid(True)
        ax.set_title(titles[i])  
        
        if i == 0:
            ax.legend()
        
    plt.suptitle(title_prefix, x=SUP_TITLE_X, ha=SUP_TITLE_HA, fontsize=SUP_TITLE_FONTSIZE)
    plt.tight_layout()
    plt.show()




def visualize_network_state(results_df_, iteration, only_flow_edges=False):

    if iteration > len(results_df_) - 1:
        raise ValueError("Too large iteration number. Max iteration number is " + str(len(results_df_) - 1))

    results_df = results_df_.copy()

    g_network_state = results_df.network_state.iloc[iteration]
    g_flow_dict = results_df.flow_dict.iloc[iteration]
    g_removed_entity = results_df.removed_entity.iloc[iteration]
    g_sources = results_df.sources.iloc[iteration]
    g_sinks = results_df.sinks.iloc[iteration]
    g_heuristic = results_df.heuristic.iloc[iteration]
    g_entity_data = results_df.entity_data.iloc[iteration]

    pos = nx.get_node_attributes(g_network_state, 'pos')
    
    europe_map = mpimg.imread('Europe_blank_map.png')
    # Use plt.imshow to display the background map
    plt.figure(figsize=(15, 10))
    plt.imshow(europe_map, extent=[-20, 40, 35, 70], alpha=0.5)

    # Extract all edges and flow edges to visualize
    all_edges_to_visualize = [(u, v) for u, v in g_network_state.edges if not g_network_state.nodes[v]['is_country_node'] and not g_network_state.nodes[u]['is_country_node']]

    flow_edges = [(u, v) for u in g_flow_dict for v in g_flow_dict[u] if g_flow_dict[u][v] > 0]
    flow_edges_to_visualize = flow_edges.copy()
    for (u, v) in flow_edges:
        if u == 'super_source' or v == 'super_sink':
            flow_edges_to_visualize.remove((u, v))

    flow_edges_to_visualize = [e for e in flow_edges if e in g_network_state.edges]

    if not only_flow_edges:
        nx.draw(g_network_state,
                pos=pos,
                with_labels=False,
                node_size=70,
                node_color=['red' if node in g_sources else 'yellow' if node in g_sinks else 'lightblue' for node in g_network_state.nodes],
                font_size=8,
                font_color="black",
                font_weight="bold",
                arrowsize=10,
                edge_color='gray',
                edgelist=all_edges_to_visualize,
                alpha=0.7)
    
    if not isinstance(g_removed_entity, tuple):  
        node_pos = {g_removed_entity: g_entity_data['pos']}
        nx.draw_networkx_nodes(g_network_state, pos=node_pos, nodelist=[g_removed_entity], node_color='blue', node_size=70)

    else:  
        nx.draw_networkx_edges(results_df.network_state.iloc[iteration-1], pos=pos, edgelist=[g_removed_entity], edge_color='blue', width=4)

    # Create a new g_network_state with only relevant edges
    nx.draw_networkx_edges(g_network_state, pos=pos, edgelist=flow_edges_to_visualize, edge_color='green', width=2)

    plt.legend(handles=[
        Line2D([0], [0], color='gray', label='Gray edges: pipelines with zero flow'),
        Line2D([0], [0], color='green', label='Green edges: pipelines with non-zero flow'),
        Line2D([0], [0], marker='o', color='blue', label='Blue entities: entity removed at current iteration')
    ], loc='lower right')

    _entity = 'node' if not isinstance(g_removed_entity, tuple) else 'edge'
    plt.suptitle('Network state at iteration ' + str(iteration)+' of '+g_heuristic+ ' heuristc, '+_entity+' removal', fontsize=20)
    plt.title('Sources: '+str(g_sources)+', sinks: '+str(g_sinks)+'\nCurrent max flow: ' + str(round(results_df.max_flow_value.iloc[iteration], 2))+ ' ['+str(round(results_df.max_flow_value.iloc[0],2))+']' +'\nCurrent flow capacity robustness: '+str(round(results_df.capacity_robustness_max_flow.iloc[iteration], 2)), fontsize=16, loc='left', y=0.95)

    if only_flow_edges:
        plt.title('Only flow edges are visualized', fontsize=16, loc='right')

    plt.show()

#------------------------------------------------------------FROM HERE ONWARDS IS CODE FROM PROJECT THESIS------------------------------------------------------------

def n_minus_k(G_, heuristic, remove, n_benchmarks=20, k_removals=250, best_worst_case=False, er_best_worst=False, print_output=False, SEED=42):

    G = G_.copy()

    results_df = pd.DataFrame(columns=['iteration', 'removed_entity', 'composite', 'robustness', 'reach', 'connectivity', 'composite_b', 'robustness_b', 'reach_b', 'connectivity_b'])
    results_best_worst_df = pd.DataFrame(columns=['iteration', 'best_entity', 'composite_best', 'worst_entity', 'composite_worst'])

    def assess_grid_connectedness_init(G):
        weakly_connected = list(nx.weakly_connected_components(G))
        largest_component = max(weakly_connected, key=len)
        largest_component_graph = G.subgraph(largest_component)

        strongly_connected = list(nx.strongly_connected_components(G))
        largest_strongly_connected = max(strongly_connected, key=len)
        largest_strongly_connected_graph = G.subgraph(largest_strongly_connected)

        return len(largest_component), len(weakly_connected), len(strongly_connected), nx.average_shortest_path_length(largest_strongly_connected_graph), nx.diameter(largest_component_graph.to_undirected()), np.average(np.array(list(list(dict(CCI(G)).values()))))
    
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

    for i in tqdm(range(0, k_removals + 1), desc='N-k iterations'):  # Loop through k_removals + 1 iterations
        b_connectedness_lst, b_robustness_lst, b_reach_lst, b_connectivity_lst = [], [], [], []
        r_connectedness_lst, r_robustness_lst, r_reach_lst, r_connectivity_lst = [], [], [], []

        # For the first iteration, calculate and store initial connectedness indices
        if i == 0:  
            composite, robustness, reach, connectivity = GCI(G, G_init, lcs_G_init, nwc_G_init, nsc_G_init, aspl_G_init, dia_G_init, comp_centrs_G_init)
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
            func = CCI_v if remove == 'node' else CCI_e
            target = func(G)

            if print_output:
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

            target = target if remove == 'node' else set(target)
            results_df.loc[i] = [i, target, composite, robustness, reach, connectivity, sum(b_connectedness_lst) / len(b_connectedness_lst), sum(b_robustness_lst) / len(b_robustness_lst), sum(b_reach_lst) / len(b_reach_lst), sum(b_connectivity_lst) / len(b_connectivity_lst)]

    return results_df, results_best_worst_df

def GCI(G, G_init, lcs_G_init, nwc_G_init, nsc_G_init, aspl_G_init, dia_G_init, comp_centr_G_init):
    largest_component_size, num_weakly_connected, num_strongly_connected, average_shortest_path_length, diameter, node_composite_centrality = get_connectedness_metrics_of(G)

    if diameter == 0:
        diameter = 1

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

    # CHANGE FROM PROJECT THESIS: instead of multiplying the indices, we take the geometric mean of the indices

    """
    GCI_reach = ((average_shortest_path_length / aspl_G_init) * ((G.number_of_edges() / diameter) / (G_init.number_of_edges() / dia_G_init)))**(1/3)


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
    largest_component = max(weakly_connected, key=len, default=[])
    largest_component_size = len(largest_component)
    num_weakly_connected = len(weakly_connected)
    num_strongly_connected = len(list(nx.strongly_connected_components(G)))

    strongly_connected = list(nx.strongly_connected_components(G))
    largest_strongly_connected = max(strongly_connected, key=len, default=[])
    largest_strongly_connected_graph = G.subgraph(largest_strongly_connected)

    if largest_component_size > 1:
        largest_component_graph = G.subgraph(largest_component)
        average_shortest_path_length = nx.average_shortest_path_length(largest_strongly_connected_graph)
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


def ER_benchmark_with_capacity(G):
    """
    Generates and returns a random directed graph using the Erdős-Rényi model with edge capacities. 
    """
    er_graph = ER_benchmark()

    for edge in er_graph.edges:
        er_graph.edges[edge]['capacity'] = np.mean([G.edges[edge]['capacity'] for edge in G.edges])
    
    return er_graph

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


#------------------------------------------------------------FROM HERE ONWARDS ARE FUNCTIONS FOR RESULTS ANALYSIS------------------------------------------------------------


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

    print("Summary statistics (first 250 removals)")
    print('----------------------------------------------')

    # Calculate the percentage loss of total max flow based on first and last row
    initial_max_flow = df_.loc[0, metric]
    final_max_flow = df_.loc[df_.index[-1], metric]
    print(f"Percentage network damage: {round(((initial_max_flow - final_max_flow) / initial_max_flow) * 100, 1)}%")
    print(f"Mean damage per entity removal: {round(df.head(250)['diff'].mean(), 2)}")
    print(f"Variation in damage per entity removal: {round(df.head(250)['diff'].std(), 2)}")
    if not pd.isna(zero_metric_iteration):
        print(f"The metric reaches 0 at iteration {zero_metric_iteration}.")


    


