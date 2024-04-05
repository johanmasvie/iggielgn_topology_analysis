from matplotlib import pyplot as plt
import numpy as np
import sympy as sp
import pandas as pd
import ast
import pickle

with open('graph_objects/G_simple_directed.pickle', 'rb') as f:
    G_simple_directed = pickle.load(f)
    G_simple_directed.name = 'G_simple_directed'

NUM_NODES_IN_G_SIMPLE_DIRECTED = G_simple_directed.number_of_nodes()
NUM_EDGES_IN_G_SIMPLE_DIRECTED = G_simple_directed.number_of_edges()


def plot_metrics(df1, df2, index1, index2):
    min_length = min(len(df1), len(df2))
    df1 = df1.iloc[:min_length]
    df2 = df2.iloc[:min_length]

    plt.figure(figsize=(10, 6))
    plt.plot(df1.index, df1[index1], label=f'{index1}')
    plt.plot(df2.index, df2[index2], label=f'{index2}')

    # Add vertical line at intersection points
    intersection_points = np.argwhere(np.diff(np.sign(df1[index1] - df2[index2]))).flatten()
    for point in intersection_points:
        if point != 0:
            plt.axvline(x=point, color='r', linestyle='--', alpha=0.4)
            plt.text(point + 1, 0.9, 'k=' + str(point), rotation=0, alpha=0.4, color='r', verticalalignment='top')
  
    # Calculate absolute difference between the two curves
    plt.fill_between(df1.index, 0, np.abs(df1[index1] - df2[index2]), color='grey', alpha=0.1)

    
    # Calculate and print AUC for each series
    auc1 = round(np.trapz(df1[index1], df1.index), 2)
    auc2 = round(np.trapz(df2[index2], df2.index), 2)
    print(f"AUC for {index1}: {auc1}")
    print(f"AUC for {index2}: {auc2}", end='\n\n')

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

    print(f"{index1}: [max ROC: {max_curvature1}, avg ROC: {average_curvature1}]")
    print(f"{index2}: [max ROC: {max_curvature2}, avg ROC: {average_curvature2}]", end='\n\n')

    plt.xlabel('k iterations')
    plt.ylabel('index')
    plt.title('Network performance index versus k entity removals')
    plt.legend()
    plt.show()


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
    


    # Calculate and print the variance of the diffs list
    print(f"Variance of index differences of common entities: {round(np.var(index_diffs),0)}")

    return result_df


  

