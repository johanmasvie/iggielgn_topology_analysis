from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import sympy as sp
import pandas as pd
import ast
import pickle
import random

SUP_TITLE_X, SUP_TITLE_HA, SUP_TITLE_FONTSIZE = 0.285, 'center', 16
SUB_PLOTS_FIGSIZE, SUB_LOC, SUB_TITLE_FONTSIZE = (20, 6), 'left', 14

SUP_TITLE_X_PLOT_TRANSLATED_ANALYSIS = 0.37

with open('graph_objects/G_simple_directed.pickle', 'rb') as f:
    G_simple_directed = pickle.load(f)
    G_simple_directed.name = 'G_simple_directed'

NUM_NODES_IN_G_SIMPLE_DIRECTED = G_simple_directed.number_of_nodes()
NUM_EDGES_IN_G_SIMPLE_DIRECTED = G_simple_directed.number_of_edges()


import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

def plot_transform_analysis(df1_pair, df2_pair, index):
    fig, axs = plt.subplots(1, 2, figsize=SUB_PLOTS_FIGSIZE)

    title = 'Centrality based N-k analysis employing greedy entity removal order resulting from max flow analysis'
    if 'max_flow_value'in df1_pair[0].columns.values:
        title = 'Max flow based N-k analysis employing greedy entity removal order resulting from centrality analysis'

    for ax, df1, df2 in zip(axs, df1_pair, df2_pair):
        min_length = min(len(df1), len(df2))
        df1 = df1.iloc[:min_length]
        df2 = df2.iloc[:min_length]

        entity_type = 'edge' if isinstance(df2.iloc[-1]['removed_entity'], set) else 'node'


        ax.set_title(f'{entity_type} removal', loc=SUB_LOC, fontsize=SUB_TITLE_FONTSIZE)
        ax.plot(df1.index, df1[index], label='transformed', color='blue')
        ax.plot(df2.index, df2[index], label='original', color='gray', alpha=0.4)

        # Add vertical line at intersection points
        intersection_points = np.argwhere(np.diff(np.sign(df1[index] - df2[index]))).flatten()
        for point in intersection_points:
            if point != 0:
                ax.axvline(x=point, color='r', linestyle='--', alpha=0.4)
                ax.text(point + 2, 0.9, 'k=' + str(point), rotation=0, alpha=0.4, color='r', verticalalignment='top')
        
        # Calculate absolute difference between the two curves
        absolute_diff = np.abs(df1[index] - df2[index])[:min_length]  # Limiting to the length of the shorter dataframe
        ax.fill_between(df1.index[:min_length], 0, absolute_diff, color='grey', alpha=0.1)
        ax.set_ylabel(index)
        ax.set_xlabel('k '+entity_type+' removals')

    # Creating a separate legend for the intersection and absolute difference
    intersection_legend = plt.Line2D([0], [0], color='r', linestyle='--', alpha=0.4, label='intersection')
    abs_diff_legend = Patch(facecolor='grey', edgecolor='black', alpha=0.1, label='absolute diff')
    fig.legend(handles=[intersection_legend, abs_diff_legend], loc='upper right', bbox_to_anchor=(0.8, 1.0))

    # Combine handles and labels for plot lines
    handles1, labels1 = axs[0].get_legend_handles_labels()
    handles2, labels2 = axs[1].get_legend_handles_labels()

    # Combine handles and labels, filtering out duplicates
    handles = handles1 + handles2
    labels = labels1 + labels2
    unique_labels = []
    unique_handles = []

    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    fig.legend(unique_handles, unique_labels, loc='upper right', bbox_to_anchor=(0.9, 1.0))

    plt.suptitle(title, x=SUP_TITLE_X_PLOT_TRANSLATED_ANALYSIS, ha=SUP_TITLE_HA, fontsize=SUP_TITLE_FONTSIZE)
    plt.show()




def plot_max_flow_and_centrality_comparison(df1_pair, df2_pair, index1, index2):
    
    fig, axs = plt.subplots(1, 2, figsize=SUB_PLOTS_FIGSIZE)
    entity_type = 'node' if ',' not in str(df2_pair[0].iloc[-1]['removed_entity']) else 'edge'


    for ax, df1, df2 in zip(axs, df1_pair, df2_pair):
        min_length = min(len(df1), len(df2))
        df1 = df1.iloc[:min_length]
        df2 = df2.iloc[:min_length]

        heuristic = 'random' if 'random' in str(df1.iloc[-1]['heuristic']) else 'greedy'

        ax.set_title(f'{heuristic} heuristic', loc=SUB_LOC, fontsize=SUB_TITLE_FONTSIZE)
        ax.plot(df1.index, df1[index1], label=f'{index1}')
        ax.plot(df2.index, df2[index2], label=f'{index2}')

        # Add vertical line at intersection points
        intersection_points = np.argwhere(np.diff(np.sign(df1[index1] - df2[index2]))).flatten()
        for point in intersection_points:
            if point != 0:
                ax.axvline(x=point, color='r', linestyle='--', alpha=0.4)
                ax.text(point + 2, 0.9, 'k=' + str(point), rotation=0, alpha=0.4, color='r', verticalalignment='top')
        
        # Calculate absolute difference between the two curves
        ax.fill_between(df1.index, 0, np.abs(df1[index1] - df2[index2]), color='grey', alpha=0.1)
        ax.set_xlabel('k '+entity_type+' removals')

    # Creating a separate legend for the intersection and absolute difference
    intersection_legend = plt.Line2D([0], [0], color='r', linestyle='--', alpha=0.4, label='intersection')
    abs_diff_legend = Patch(facecolor='grey', edgecolor='black', alpha=0.1, label='absolute diff')
    fig.legend(handles=[intersection_legend, abs_diff_legend], loc='upper right', bbox_to_anchor=(0.75, 1.0))

    # Combine handles and labels for plot lines
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.9, 1.0))

    plt.suptitle('Comparison of max flow and centrality based N-k anlyses', x=SUP_TITLE_X, ha=SUP_TITLE_HA, fontsize=SUP_TITLE_FONTSIZE)
    plt.show()
    
    for df1, df2 in zip(df1_pair, df2_pair):
        min_length = min(len(df1), len(df2))
        df1 = df1.iloc[:min_length]
        df2 = df2.iloc[:min_length]

        # Calculate and print AUC for each series
        auc1 = round(np.trapz(df1[index1], df1.index), 2)
        auc2 = round(np.trapz(df2[index2], df2.index), 2)

        # Fit a polynomial of degree 2 (change this to fit a polynomial of a different degree)
        coeffs1 = np.polyfit(df1.index, df1[index1], 3)
        coeffs2 = np.polyfit(df2.index, df2[index2], 3)

        # Calculate curvature for polynomial fits
        x = sp.Symbol('x')
        f1 = coeffs1[0]*x**2 + coeffs1[1]*x**2 + coeffs1[2]*x + coeffs1[3]
        f2 = coeffs2[0]*x**2 + coeffs2[1]*x**2 + coeffs2[2]*x + coeffs2[3]

        f1_prime, f2_prime = sp.diff(f1, x), sp.diff(f2, x)
        f1_double_prime, f2_double_prime = sp.diff(f1_prime, x), sp.diff(f2_prime, x)

        curvature_formula1 = sp.Abs(f1_double_prime) / (1 + f1_prime**2)**(3/2)
        curvature_formula2 = sp.Abs(f2_double_prime) / (1 + f2_prime**2)**(3/2)

        interval_start, interval_end = min(df1.index[0], df2.index[0]), max(df1.index[min_length-1], df2.index[min_length-1])
        interval_points = np.linspace(interval_start, interval_end, 100)
        curvature_values1, curvature_values2 = [curvature_formula1.subs(x, point) for point in interval_points], [curvature_formula2.subs(x, point) for point in interval_points]

        # Calculate max curvature and average curvature
        max_curvature1, max_curvature2 = round(max(curvature_values1) * 1000, 5), round(max(curvature_values2) * 1000, 5)
        average_curvature1,average_curvature2= round(np.mean(curvature_values1) * 1000, 5), round(np.mean(curvature_values2) * 1000, 5)

        # Print AUC and ROC information side by side
        print(f"\tAUC for {index1}: {auc1}  \t\t\t\t\t\t\t          AUC for {index2}: {auc2}")
        print(f"\t{index1}: [max ROC: {max_curvature1}, avg ROC: {average_curvature1}]    \t\t\t\t\t  {index2}: [max ROC: {max_curvature2}, avg ROC: {average_curvature2}]\n")

    



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

def fix_centrality_edge(edges_lst):
    new_edges_lst = []
    for e in edges_lst:
        if e in G_simple_directed.edges():
            new_edges_lst.append(e)
        reversed_e = (e[1], e[0])
        if reversed_e in G_simple_directed.edges():
            if reversed_e not in new_edges_lst:
                new_edges_lst.append(reversed_e)
    return list(set(new_edges_lst))
