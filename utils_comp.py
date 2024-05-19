from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import sympy as sp
import pandas as pd
import ast
import pickle
import random

SUP_TITLE_X, SUP_TITLE_HA, SUP_TITLE_FONTSIZE = 0.288, 'center', 16
SUB_PLOTS_FIGSIZE, SUB_LOC, SUB_TITLE_FONTSIZE = (20, 6), 'left', 14

SUP_TITLE_X_PLOT_TRANSLATED_ANALYSIS = 0.37

with open('graph_objects/G_simple_directed_iggielgn.pickle', 'rb') as f:
    G_simple_directed = pickle.load(f)
    G_simple_directed.name = 'G_simple_directed'

NUM_NODES_IN_G_SIMPLE_DIRECTED = G_simple_directed.number_of_nodes()
NUM_EDGES_IN_G_SIMPLE_DIRECTED = G_simple_directed.number_of_edges()


import numpy as np
def plot_transform_comparison(df1, df2, df3, metric):
    """
    Plots the specified performance metric against the number of iterations for three dataframes.
    
    Parameters:
    df1 (pd.DataFrame): Tranformed data
    df2 (pd.DataFrame): Original data, greedy
    df3 (pd.DataFrame): Original data, random
    metric (str): The column name of the performance metric to plot.
    
    Returns:
    None
    """
    
    # Determine the minimum length of the dataframes
    min_length = min(len(df1), len(df2), len(df3))
    
    # Truncate dataframes to the minimum length
    df1_truncated = df1.head(min_length)
    df2_truncated = df2.head(min_length)
    df3_truncated = df3.head(min_length)

    # Calculate the differences for shading and area calculation
    diff1_vs_2 = abs(df1_truncated[metric] - df2_truncated[metric])
    diff1_vs_3 = abs(df1_truncated[metric] - df3_truncated[metric])
    
    # Calculate the areas of the shaded regions
    area1_vs_2 = round(np.trapz(diff1_vs_2.clip(lower=0), dx=1),1)
    area1_vs_3 = round(np.trapz(diff1_vs_3.clip(lower=0), dx=1),1)

    # Calculate positions for annotations
    mean_y1_vs_2 = (df1_truncated[metric] + df2_truncated[metric]) / 2
    mean_y1_vs_3 = (df1_truncated[metric] + df3_truncated[metric]) / 2
    
    idx1_vs_2 = np.argmax(diff1_vs_2)
    idx1_vs_3 = np.argmax(diff1_vs_3)
    
    # Plot the performance metric against the number of iterations
    fig = plt.figure(figsize=(10, 6))
    
    plt.plot(df1_truncated.index, df1_truncated[metric], label='transformed data', color='blue')
    plt.plot(df2_truncated.index, df2_truncated[metric], label='original data, greedy', color='gray')
    plt.plot(df3_truncated.index, df3_truncated[metric], label='original data, random', color='gray', linestyle='--')
    
    # Add shading between df1 and df2
    plt.fill_between(df1_truncated.index, df1_truncated[metric], df2_truncated[metric], 
                     # where=(df1_truncated[metric] > df2_truncated[metric]), 
                     interpolate=True, color='lightgray', alpha=0.3)
    
    # Add shading between df1 and df3
    plt.fill_between(df1_truncated.index, df1_truncated[metric], df3_truncated[metric], 
                     # where=(df3_truncated[metric] > df1_truncated[metric]), 
                     interpolate=True, color='lightblue', alpha=0.1, linestyle='--')
    
    # Annotate the areas of the shaded regions
    plt.annotate(f'Area: {area1_vs_2}', 
                xy=(idx1_vs_2, mean_y1_vs_2.iloc[idx1_vs_2]), 
                xytext=(idx1_vs_2, mean_y1_vs_2.iloc[idx1_vs_2] - 0.1),
                fontsize=12, backgroundcolor='white', alpha=0.5)
    
    plt.annotate(f'Area: {area1_vs_3}', 
                xy=(idx1_vs_3, mean_y1_vs_3.iloc[idx1_vs_3]), 
                xytext=(idx1_vs_3, mean_y1_vs_3.iloc[idx1_vs_3] - 0.05),
                fontsize=12, backgroundcolor='white', alpha=0.5)
    
    metric_txt = 'NPI' if metric == 'NPI' else 'FCR'
    plt.ylabel(metric_txt, fontsize=15)
   
    remove_txt = 'edge' if len(df2_truncated.iloc[1]['removed_entity']) == 2 else 'node'
    plt.xlabel('k ' + remove_txt + ' removals', fontsize=15)

    # Set size of y and x-axis ticks to 12
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Set size of x and y-axis labels to 15
    plt.ylabel(metric_txt, fontsize=15)
    
    if remove_txt == 'node':
        plt.legend(fontsize=12, loc='upper right')
        
    plt.grid(True, alpha=0.2)
    plt.show()
    return fig


def plot_direct_comparison(df1, df2, df3, df4):
    """
    Plots the performance metrics from four dataframes.
    
    Parameters:
    df1, df2: DataFrames containing NPI metrics, random and greedy 
    df3, df4: DataFrames containing FCR metrics, random and greedy
    
    Each DataFrame is expected to have the performance metric as a column named 'NPI' or 'FCR'.
    The index of the DataFrame is used as the x-axis (iterations).
    """
    
    # Create the plot
    plt.figure(figsize=(10, 5))

    # Ensure dataframes have the same length
    min_length = min(len(df1), len(df2), len(df3), len(df4))
    df1, df2, df3, df4 = df1.iloc[:min_length], df2.iloc[:min_length], df3.iloc[:min_length], df4.iloc[:min_length]

    # Determine entity type based on the last row of df1
    entity_type = 'edge' if isinstance(df1.iloc[-1]['removed_entity'], set) else 'node'

    
    # Plot NPI data
    plt.plot(df1.index, df1['NPI'], label='NPI, greedy', linestyle='-', color='blue')
    plt.plot(df2.index, df2['NPI'], label='NPI, random', linestyle='--', color='lightblue')
    
    # Plot FCR data
    plt.plot(df3.index, df3['capacity_robustness_max_flow'], label='FCR, greedy', linestyle='-', color='coral')
    plt.plot(df4.index, df4['capacity_robustness_max_flow'], label='FCR, random', linestyle='--', color='lightcoral')
    
    # Adding labels and title
    plt.xlabel('k ' + entity_type + ' removals', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.2)
    
    plt.savefig('saved_plots/iggielgn/hybrid/comparison/direct_'+entity_type+'_comparison.png', bbox_inches='tight', pad_inches=0)
    
    print_AUC_ROC_info(df1)
    print_AUC_ROC_info(df2)
    print_AUC_ROC_info(df3)
    print_AUC_ROC_info(df4)

    # Increase the size of ticks
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.tick_params(axis='both', which='minor', labelsize=10)

    plt.show()

def print_AUC_ROC_info(df):
    """
    Calculates and prints AUC and ROC information for a single dataframe.

    Parameters:
    df (DataFrame): The dataframe containing the data.
    index (str): The column name of the performance metric in the dataframe.
    """
    # Ensure the dataframe has a sufficient length
    min_length = len(df)
    df = df.iloc[:min_length]

    index = 'NPI' if 'NPI' in df.columns else 'capacity_robustness_max_flow'

    # Calculate AUC
    auc = round(np.trapz(df[index], df.index), 2)

    # Fit a polynomial of degree 3
    coeffs = np.polyfit(df.index, df[index], 3)

    # Calculate curvature for polynomial fit
    x = sp.Symbol('x')
    f = coeffs[0]*x**3 + coeffs[1]*x**2 + coeffs[2]*x + coeffs[3]

    f_prime = sp.diff(f, x)
    f_double_prime = sp.diff(f_prime, x)

    curvature_formula = sp.Abs(f_double_prime) / (1 + f_prime**2)**(3/2)

    interval_start, interval_end = df.index[0], df.index[min_length-1]
    interval_points = np.linspace(interval_start, interval_end, 100)
    curvature_values = [curvature_formula.subs(x, point) for point in interval_points]

    # Calculate max curvature and average curvature
    max_curvature = round(max(curvature_values) * 1000, 5)
    average_curvature = round(np.mean(curvature_values) * 1000, 5)

    # Print AUC and ROC information
    print(f"AUC for {index}: {auc}")
    print(f"{index}: [max ROC: {max_curvature}, avg ROC: {average_curvature}]\n")


def common_entities(df1_, df2_):
    # Copy dataframes to avoid modifying the original data
    df1, df2 = df1_.copy(), df2_.copy()

    # Remove the first row from both dataframes which contains the initial state
    df1, df2 = df1.iloc[1:], df2.iloc[1:]

    # Determine the common domain (iterations) where removal process stops
    common_domain = min(df1.index.max(), df2.index.max())

    # Create an empty list to store index differences for all entities
    index_diffs = []

    if isinstance(df1.iloc[0]['removed_entity'], set):
        df1['removed_entity'] = df1['removed_entity'].apply(lambda x: str(list(x)))
        df2['removed_entity'] = df2['removed_entity'].apply(lambda x: str(list(x)))

    # Get unique removed entities present in both dataframes
    common_entities = set(df1['removed_entity'].unique()).intersection(set(df2['removed_entity'].unique()))

    # Create an empty list to store dictionaries with the data
    result_data = []

    # Iterate over common removed entities
    for removed_entity in common_entities:
        # Get the index positions of the removed entity in both dataframes
        df1_indices = df1[df1['removed_entity'] == removed_entity].index
        df2_indices = df2[df2['removed_entity'] == removed_entity].index

        # Filter out index positions beyond the common domain
        df1_indices = df1_indices[df1_indices <= common_domain]
        df2_indices = df2_indices[df2_indices <= common_domain]

        # Ensure that both dataframes have index positions
        if len(df1_indices) > 0 and len(df2_indices) > 0:
            # Calculate the differences in index positions
            diffs = df2_indices - df1_indices

            # Append the differences to the list
            index_diffs.extend(diffs)

            # Add the data to the result list
            for idx in range(min(len(df1_indices), len(df2_indices))):
                result_data.append({'removed_entity': removed_entity,
                                    'k_iteration [centrality]': df1_indices[idx],
                                    'k_iteration [max_flow]': df2_indices[idx],
                                    'diff': diffs[idx]})
                
    # Create DataFrame from the list of dictionaries
    result_df = pd.DataFrame(result_data)
    if '[' in df1.iloc[0]['removed_entity']:
        result_df['removed_entity'] = result_df['removed_entity'].apply(lambda x: tuple(set(ast.literal_eval(x))))
    print(f"{len(result_df)} common entity removals")
    print(f"{len(df2)} greedy entity removals before '{df2.iloc[0]['heuristic']}' reached 0 (limiting metric)\n")

    def calculate_variance(entities, num_entities_df1, num_entities_df2):
        """
        Calculate the variance of the differences in index positions of unique numbers in two lists of entities. 
        """
        total_variance = 0
        num_iterations = 100
        for _ in range(num_iterations):
            # Define a sequence of unique numbers in the range of entities
            unique_numbers = list(range(entities))

            # Generate two lists, one for df1 and one for df2
            df1_list = random.sample(unique_numbers, min(entities, num_entities_df1))
            df2_list = random.sample(unique_numbers, min(entities, num_entities_df2))

            # Calculate differences in index positions of unique numbers
            differences = []
            for num in unique_numbers:
                if num in df1_list and num in df2_list:
                    diff = abs(df1_list.index(num) - df2_list.index(num))
                    differences.append(diff)

            # Calculate variance
            if differences:  # Check if differences list is not empty
                variance = sum((x - (sum(differences) / len(differences))) ** 2 for x in differences) / len(differences)
                total_variance += variance

        average_variance = total_variance / num_iterations if total_variance > 0 else 0  # Handle division by zero
        return round(average_variance, 0)

    # Calculate and print the variance of the diffs list
    print(f"Variance of index differences of common entities: {round(np.var(index_diffs),0)}")
    if '[' in df1.iloc[0]['removed_entity']:
        print(f"Variance of index differences for random (averaged) edge removal: {calculate_variance(NUM_EDGES_IN_G_SIMPLE_DIRECTED, len(df1), len(df2))}")
    else:
        print(f"Variance of index differences for random (averaged) node removal: {calculate_variance(NUM_NODES_IN_G_SIMPLE_DIRECTED, len(df1), len(df2))}")

    # Take the absolute value of the 'diff' column
    result_df['abs_diff'] = result_df['diff'].abs()
    result_df.drop(columns='diff', inplace=True)

    # Sort the DataFrame by 'diff' in ascending order
    result_df = result_df.sort_values(by='abs_diff')

    return result_df

def correct_edges(edges_lst):
    new_edges_lst = []
    seen = set()
    for e in edges_lst:
        if e in G_simple_directed.edges():
            if e not in seen:
                new_edges_lst.append(e)
                seen.add(e)
        else:
            reversed_e = (e[1], e[0])
            if reversed_e in G_simple_directed.edges():
                if reversed_e not in seen:
                    new_edges_lst.append(reversed_e)
                    seen.add(reversed_e)
    return new_edges_lst
