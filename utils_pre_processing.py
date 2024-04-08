import pandas as pd
import json
import ast

def expand_dict_column(row):
    # Initialize an empty dict to hold the expanded data
    expanded_data = {}
    for key, value in row.items():
        if isinstance(value, list):
            # Convert list to a string representation
            expanded_data[key] = str(value)
        else:
            # Directly assign non-list values
            expanded_data[key] = value
    return pd.Series(expanded_data)

def split_column_to_multiple(df, column, prefix=None):
    df[column] = df[column].apply(lambda x: ast.literal_eval(x))
    expanded_columns = df[column].apply(expand_dict_column).apply(pd.Series)
    if prefix:
        expanded_columns = expanded_columns.add_prefix(prefix)
    df = df.join(expanded_columns)
    df.drop(column, axis=1, inplace=True)
    return df

def split_col_to_and_from(df, col, type):
    try:
        df[col] = df[col].apply(lambda x: json.loads(x.replace("'", '"')))
        if type == 'str':
            df[col] = df[col].apply(lambda x: [str(i) for i in x])
        elif type == 'int':
            df[col] = df[col].apply(lambda x: [int(i) for i in x])
        elif type == 'float':
            df[col] = df[col].apply(lambda x: [float(i) for i in x])
        df['from_' + col] = df[col].apply(lambda x: x[0])
        df['to_' + col] = df[col].apply(lambda x: x[1])
        df = df.drop(columns=[col])
    except Exception as e:
        print(f"An error occurred while processing column '{col}': {str(e)}")
    return df

def split_coords (df, col):
    # The 'lat' and 'long' columns both on the format "[60.54204007494405, 60.54266109350547, 60.54330065370974, 60.56023487203963]"
    # We want to split this into four columns: 'from_lat', 'from_long', 'to_lat', 'to_long'
    # The first value in each column is the value of the start node, from_lat and from_long, and the last value is the end node, to_lat and to_long
    try:
        df[col] = df[col].apply(lambda x: json.loads(x.replace("'", '"')))
        df['from_' + col] = df[col].apply(lambda x: x[0])
        df['to_' + col] = df[col].apply(lambda x: x[-1])
        df = df.drop(columns=[col])
    except Exception as e:
        print(f"An error occurred while processing column '{col}': {str(e)}")
    return df