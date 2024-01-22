import pandas as pd
import numpy as np
import networkx as nx

def create_graphs_from_dataset(df):
    graphs = []
    mm_yyyy = df.iloc[:, df.columns.get_loc('Oct-08'):]

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



    


