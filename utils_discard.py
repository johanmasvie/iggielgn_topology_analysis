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


def max_flow_edge_count(G, count_or_flow='count'):
    """
    Calculates the number of times each edge is part of the max flow path between any two nodes in the graph G. 
    """

    nodes = list(G.nodes)
    n = len(nodes)

    edge_count = {}
    
    for i in range(n):

        # The current node
        source = nodes[i]

        # Skip nodes with out-degree 0
        if G.out_degree(source) == 0:
            continue        

        for j in range(n):

            # Node the max flow is calculated to
            sink = nodes[j]

            if source != sink and (source, sink) in G.edges:
            
                if nx.has_path(G, source, sink):

                    flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity='max_cap_M_m3_per_d')
                                    
                    for u, flows in flow_dict.items():
                        for v, flow in flows.items():
                            if (u, v) not in edge_count:
                                edge_count[(u, v)] = 0
                            
                            if count_or_flow == 'flow':
                                edge_count[(u, v)] += flow
                                continue

                            if flow > 0 and count_or_flow == 'count':
                                edge_count[(u, v)] += 1
                                continue

                            if flow > 0 and count_or_flow == 'load_rate':
                                edge_count[(u, v)] += (flow / G.edges[(u, v)]['max_cap_M_m3_per_d'])

                    continue          
    
    edge_count_raw = {k: v for k, v in edge_count.items()}
    
    edge_count_combined = {k: {'edge': k, 'max_flow_edge_count': v} for k, v in edge_count_raw.items()}
    
    df = pd.DataFrame.from_dict(edge_count_combined, orient='index').reset_index()

    if df.empty:
        return df
    
    df.drop(columns=['level_0', 'level_1'], inplace=True)
    
    df = df.sort_values(by='max_flow_edge_count', ascending=False)
    return df

def edge_cutset_count(G, observed_min_cutsets, current_iteration):
    """
    Calculates the number of times each edge is part of the minimum cutset between any two nodes in the graph G.

    """
    nodes = list(G.nodes)
    n = len(nodes)

    edge_cutset_count_ = {edge: 0 for edge in G.edges}

    for i in tqdm(range(n), desc='Calculating min cutset count'):

        source = nodes[i]

        if G.out_degree(source) == 0:
            continue

        for j in range(n):

            sink = nodes[j]

            if source != sink and (source, sink) in G.edges:

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
                    continue

            observed_min_cutsets[(source, sink)] = 0

    data = [{
        'edge': edge,
        'min_cutset_count': value,
    } for edge, value in edge_cutset_count_.items()]

    if not data:
        return pd.DataFrame(), observed_min_cutsets

    df = pd.DataFrame(data)
    df = df.sort_values(by='min_cutset_count', ascending=False)

    return df, observed_min_cutsets


def weighted_flow_capacity_rate(G):
    """ 
    Calculates the Weighted Flow Capacity Robustness (WFCR) of the graph G.    
    """

    nodes = list(G.nodes)
    n = len(nodes)

    edge_WFCR = {}
    tot_flow = 0

    for i in tqdm(range(n), desc='Calculating wfcr'):

        source = nodes[i]

        if G.out_degree(source) == 0:
            continue

        for j in range(n):

            sink = nodes[j]

            if source != sink and (source, sink) in G.edges:

                if nx.has_path(G, source, sink):
                    
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
            remove = 'edge' if isinstance(df.iloc[1]['removed_entity'], set) else 'node'
            ax.plot(df.index, df[series_name], marker='o', label=f'{heuristic} {remove} removals')

        ax.set_xlabel('k iterations')
        ax.set_ylabel(series_name.replace('_', ' '))
        ax.set_title(f'{series_name.replace("_", " ").title()} vs k removals')

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


    


