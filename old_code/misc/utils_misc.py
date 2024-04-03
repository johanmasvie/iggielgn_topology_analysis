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