from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import numpy as np
import networkx as nx
import random
import seaborn as sns


SEED=42
SUP_TITLE_X, SUP_TITLE_HA, SUP_TITLE_FONTSIZE = 0.12, 'center', 'x-large'
SUB_PLOTS_FIGSIZE = (12, 6)

#------------------------------------------------------------FROM HERE ONWARDS IS CODE FROM PROJECT THESIS------------------------------------------------------------

def n_minus_k(G_, heuristic, remove='node', n_benchmarks=20, k_removals=2500, greedy_max_flow_lst=None, SEED=42):

    G = G_.copy()

    results_df = pd.DataFrame(columns=['iteration', 'removed_entity', 'NPI', 'connectedness', 'reach', 'connectivity', 'heuristic'])

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
            composite, robustness, reach, connectivity, next_target_node = NPI(G, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init)

            results_df.loc[i] = [i, None, composite, robustness, reach, connectivity, None]
            continue

            
        # Rest of iterations (i.e., not i == 0)
        if heuristic == 'random':

            for real_copy in r_graphs:  # Iterate through the instantiated copies
                target = random.choice(list(real_copy.nodes() if remove == 'node' else real_copy.edges()))

                real_copy.remove_node(target) if remove == 'node' else real_copy.remove_edge(*target)
                r_composite, r_robustness, r_reach, r_connectivity, _ = NPI(real_copy, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init)
                
                # Log indices for each modified graph
                r_connectedness_lst.append(r_composite)
                r_robustness_lst.append(r_robustness)
                r_reach_lst.append(r_reach)
                r_connectivity_lst.append(r_connectivity)
            
            
            results_df.loc[i] = [i, None,
                                sum(r_connectedness_lst) / len(r_connectedness_lst), sum(r_robustness_lst) / len(r_robustness_lst), sum(r_reach_lst) / len(r_reach_lst), sum(r_connectivity_lst) / len(r_connectivity_lst), 'random']

        elif heuristic == 'greedy':
            # func = CCI_v if remove == 'node' else lambda G: CCI_e(G, type='Li et al., 2021')
            # target = func(G)

            target = CCI_e(G, type='Li et al., 2021') if remove == 'edge' else next_target_node

            G.remove_node(target) if remove == 'node' else G.remove_edge(*target)
            target = target if remove == 'node' else set(target)                
            composite, robustness, reach, connectivity, next_target_node = NPI(G, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init)

            results_df.loc[i] = [i, target, composite, robustness, reach, connectivity, 'greedy']

            if composite == 0:
                return results_df
                
          
        elif heuristic == 'max_flow':
            if greedy_max_flow_lst is None:
                raise ValueError("Please provide a list of greedy max flows for the 'max_flow' heuristic.")
            try:
                target = greedy_max_flow_lst[i-1] # Since the first iteration i == 0 is assigned None
            except IndexError:
                return results_df
            
            if isinstance(target, tuple):
                if target not in G.edges():
                    target = tuple([target[1], target[0]])
                    if target not in G.edges():
                        print(f"Edge {target} not in graph. Skipping...")
                        continue
    
            G.remove_edge(*target) if isinstance(target, tuple) else G.remove_node(target)
            target = target if remove == 'node' else set(target)  
            composite, robustness, reach, connectivity, _ = NPI(G, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centrs_G_init)
            results_df.loc[i] = [i, target, composite, robustness, reach, connectivity, 'max_flow']
            continue

            

    return results_df

def NPI(G, lcs_G_init, nwc_G_init, nsc_G_init, dia_G_init, comp_centr_G_init):
    largest_component_size, num_weakly_connected, num_strongly_connected, diameter, node_composite_centrality, next_target_node = get_connectedness_metrics_of(G)

    """
    CONNECTEDNESS 
    Measures the network's resilience against random failures or targeted attacks. 
    It's approximated by the relative sizes of the largest connected component, weakly connected components, and strongly connected components.

    """

        
    NPI_connectedness = (largest_component_size / lcs_G_init) * (nwc_G_init / num_weakly_connected) * ( nsc_G_init / num_strongly_connected)

    """
    REACH 
    Quantifies the efficiency of information flow or reachability across the network. 
    It's reflected in the ratio of current average shortest path length and network diameter in comparison to their initial values.
    """
    NPI_reach = (diameter / dia_G_init)

    """
    CONNECTIVITY 
    Represents the centrality or average connectivity of the network. 
    It's estimated by comparing the current median node degree to the initial median node degree.
    """
    NPI_connectivity = (node_composite_centrality / comp_centr_G_init)

    """
    Calculate the composite connectedness index using the above calculated indices
    """
    NPI = NPI_connectedness * NPI_reach * NPI_connectivity
    
    return NPI, NPI_connectedness, NPI_reach, NPI_connectivity, next_target_node

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

    # node_composite_centrality = np.average(np.array(list(dict(CCI(G.subgraph(largest_component))).values())))

    CCI_dict = CCI(G.subgraph(largest_component))
    node_composite_centrality = np.average(np.array(list(dict(CCI_dict).values())))
    next_target_node = max(CCI_dict, key=CCI_dict.get)

    return largest_component_size, num_weakly_connected, num_strongly_connected, diameter, node_composite_centrality, next_target_node


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
    
def plot_comparison(greedy_df, random_df, entity='node'):
    # Create figure for the first plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))

    # Define the range of data to be displayed in the plot
    plot_range = slice(0, 1000)  # Display first 1000 iterations

    # Plot 'robustness', 'reach', and 'connectivity'
    greedy_colors = ['blue', 'green', 'red']
    random_colors = ['lightblue', 'lightgreen', 'salmon']
    metrics = ['connectedness', 'reach', 'connectivity']
    labels = ['Greedy', 'Random']
    
    marker_styles = {
        'connectedness': 's',  # square
        'reach': '^',            # triangle
        'connectivity': 'o'     # circle
    }

    for metric, greedy_color, random_color in zip(metrics, greedy_colors, random_colors):
        marker = marker_styles.get(metric, 'o')  
        ax1.plot(greedy_df['iteration'].loc[plot_range], greedy_df[metric].loc[plot_range], marker=marker, color=greedy_color, label=f'{metric}, {labels[0].lower()}', linewidth=1, markersize=5, markevery=25)
        ax1.plot(random_df['iteration'].loc[plot_range], random_df[metric].loc[plot_range], '--', marker=marker, color=random_color, label=f'{metric}, {labels[1].lower()}', linewidth=1, markersize=5, markevery=25)

    ax1.set_xlabel('k '+entity+' removals', fontsize=20)
    ax1.legend(loc='upper right', fontsize=15)
    ax1.set_ylabel('Network Centrality Performance Index, NCPI', fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=15)



    # Add a text box with a description of the markers
    fig1.text(x=0.675, y=0.35, s='markers every 25 iterations', alpha=0.5,
              bbox=dict(facecolor='white', edgecolor='lightgray'), fontsize=15) 


    plt.tight_layout()
    plt.grid(True, alpha=0.2)
    plt.savefig('saved_plots/iggielgn/centrality/'+entity+'_removals_subindices.png', bbox_inches='tight', pad_inches=0.1)
    plt.show()

    # Create figure for the second plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Define the range of data to be displayed in the plot
    plot_range = slice(0, 100)  # Display first 100 iterations

    # Plot 'NPI'
    ax2.plot(greedy_df['iteration'].loc[plot_range], greedy_df['NPI'].loc[plot_range], color='blue', label=r'$\varphi^{\text{NCPI}}_{G_k}, \text{greedy}$')
    ax2.plot(random_df['iteration'].loc[plot_range], random_df['NPI'].loc[plot_range], '--', color='lightblue', label=r'$\varphi^{\text{NCPI}}_{G_k}, \text{random}$')


    ax2.set_xlabel('k '+entity+' removals', fontsize=20)
    ax2.legend(fontsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)


    plt.tight_layout()
    plt.grid(True, alpha=0.2)
    plt.savefig('saved_plots/iggielgn/centrality/'+entity+'_removals_index.png', bbox_inches='tight', pad_inches=0.1)
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

    print("Summary statistics (first 10 removals)")
    print('----------------------------------------------')

    # Calculate the percentage loss of total max flow based on first and last row
    initial_max_flow = df_.loc[0, metric]
    final_max_flow = df_.loc[df_.index[10], metric]
    print(f"Percentage network damage: {round(((initial_max_flow - final_max_flow) / initial_max_flow) * 100, 1)}%")
    print(f"Mean damage per entity removal: {round(df.head(10)['diff'].mean(), 2)}")
    print(f"Variation in damage per entity removal: {round(df.head(10)['diff'].std(), 2)}")
    if not pd.isna(zero_metric_iteration):
        print(f"The metric reaches 0 at iteration {zero_metric_iteration}.")



def average_dfs(df1, df2, metrics_to_average):
    # Find the DataFrame with the most rows
    if len(df1) >= len(df2):
        longest_df = df1.copy()
        shorter_df = df2.copy()
    else:
        longest_df = df2.copy()
        shorter_df = df1.copy()

    # Initialize a list to store the dictionaries representing rows of the resulting DataFrame
    rows = []

    # Iterate through the rows of the shorter DataFrame
    for i, row in shorter_df.iterrows():
        # Get the 'iteration' value from the shorter DataFrame
        iteration_value = row['iteration']

        # Calculate the average for each specified metric
        averaged_values = {'iteration': iteration_value}
        for metric in metrics_to_average:
            if metric in df1.columns and metric in df2.columns:
                averaged_values[metric] = (df1[metric][i] + df2[metric][i]) / 2
            elif metric in df1.columns:
                averaged_values[metric] = df1[metric][i]
            elif metric in df2.columns:
                averaged_values[metric] = df2[metric][i]

        # Add the averaged values to the list
        rows.append(averaged_values)

    # Add remaining rows from the longer DataFrame
    if len(longest_df) > len(shorter_df):
        remaining_rows = longest_df.iloc[len(shorter_df):].to_dict('records')
        rows.extend(remaining_rows)

    # Create the DataFrame from the list of dictionaries
    averaged_df = pd.DataFrame(rows)

    return averaged_df