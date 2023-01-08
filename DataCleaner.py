# Helper class cleans up data from large csv of all NFL plays

import pandas as pd


def data_clean_controller(csv_file_name: str):
    df = pd.read_csv(csv_file_name)
    scoot_columns_left(df, "play_type", "yards_gained", "shotgun")
    scoot_columns_left(df, "pass_location", "air_yards", "yards_after_catch")
    drop_nonnumeric_rows(df, "air_yards")
    drop_nonnumeric_rows(df, "yards_gained")
    df.to_csv(csv_file_name+"_cleaned.csv", index = False)


# returns list of indexes for all rows with non-numeric entries in relevant column
def get_broken_row_indexes(df: pd.DataFrame, column_title: str):
    indexes_of_rows_with_issue = []
    #get all indexes of rows with yards_gained as string that cannot be made numeric
    for idx in df.index:
        relevant_datapoint = df[column_title][idx]
        if type(relevant_datapoint) == str and not relevant_datapoint.strip('-').isnumeric():
            indexes_of_rows_with_issue.append(idx)
    return indexes_of_rows_with_issue

# in several spots in original csv, one column has been scooted right, putting string values
# into a numeric column and causing bugs. This fixes that
def scoot_columns_left(df: pd.DataFrame, left_column: str, mid_column: str, right_column:str):
    indexes_of_rows_with_issue = get_broken_row_indexes(df, mid_column)

    #shifts relevant columns over, but will not fix "shotgun" column (which contains yards_gained in bad rows)
    for idx in indexes_of_rows_with_issue:
        df[left_column][idx] = df[mid_column][idx]
        df[mid_column][idx] = df[right_column][idx]

# Checks if each entry in relevant column is a number or can be made a number.
# If not, drops entire row.
def drop_nonnumeric_rows(df: pd.DataFrame, column_name: str):
    indexes_of_bad_rows = get_broken_row_indexes(df, column_name)
    df.drop(indexes_of_bad_rows, axis=0, inplace=True)