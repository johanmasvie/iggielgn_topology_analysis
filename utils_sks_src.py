from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
import random
import matplotlib.image as mpimg


SEED=42
SUP_TITLE_X, SUP_TITLE_HA, SUP_TITLE_FONTSIZE = 0.12, 'center', 'x-large'
SUB_PLOTS_FIGSIZE = (12, 6)

###------------------------------------------------------------FROM HERE ARE ALGORITHMS AND GETTERS------------------------------------------------------------###



def max_flow_edge_count(G, global_sources_lst, count_or_flow='count'):
    """
    Calculates the number of times each edge is part of the max flow path between any two nodes in the graph G. 
    """

    nodes = list(G.nodes)
    n = len(nodes)

    edge_count = {}
    
    for i in range(n):

        # The current node
        source = nodes[i]

        if source in global_sources_lst:        

            for j in range(n):

                # Node the max flow is calculated to
                sink = nodes[j]

                if source != sink:
                
                    if nx.has_path(G, source, sink):

                        try:
                            flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity='max_cap_M_m3_per_d')
                        except IndexError:
                            continue
                                        
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

def edge_cutset_count(G, observed_min_cutsets, global_sources_lst, current_iteration):
    """
    Calculates the number of times each edge is part of the minimum cutset between any two nodes in the graph G.

    """
    nodes = list(G.nodes)
    n = len(nodes)

    edge_cutset_count_ = {edge: 0 for edge in G.edges}

    for i in tqdm(range(n), desc='Calculating min cutset count'):

        source = nodes[i]

        if source in global_sources_lst:

            for j in range(n):

                sink = nodes[j]

                if source != sink:

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


def weighted_flow_capacity_rate(G, global_sources_lst):
    """ 
    Calculates the Weighted Flow Capacity Robustness (WFCR) of the graph G.    
    """

    nodes = list(G.nodes)
    n = len(nodes)

    edge_WFCR = {}
    tot_flow = 0

    for i in tqdm(range(n), desc='Calculating wfcr'):

        source = nodes[i]

        if source in global_sources_lst:

            for j in range(n):

                sink = nodes[j]

                if source != sink:

                    if nx.has_path(G, source, sink):

                        try:
                            flow_value, flow_dict = nx.maximum_flow(G, source, sink, capacity='max_cap_M_m3_per_d')
                        except IndexError:
                            continue
                        
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


    


