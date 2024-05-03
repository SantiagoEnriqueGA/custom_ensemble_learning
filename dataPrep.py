import pandas as pd

class DataPrep(object):
    """
    A class for preparing data for machine learning models.
    """

    def one_hot_encode(df, cols):
        """
        One-hot encodes non-numerical columns in a DataFrame.

        Parameters:
        - df (pandas.DataFrame): The DataFrame to be encoded.
        - cols (list): The list of column indices to be encoded.

        Returns:
        - df (pandas.DataFrame): The DataFrame with one-hot encoded columns.
        """

        # One-hot-encode non-numerical columns
        for col in cols:
            # Get unique values in the column
            unique_values = df.iloc[:, col].unique()

            # For each unique value, create a new binary column
            for value in unique_values:
                df[str(value)] = (df.iloc[:, col] == value).astype(int)

        # Drop the original column
        df = df.drop(df.columns[cols], axis=1)

        return df

    def write_data(df, csv_file):
        """
        Writes the DataFrame to a CSV file.

        Parameters:
        - df (pandas.DataFrame): The DataFrame to be written.
        - csv_file (str): The path of the CSV file to write to.
        """

        # Write the DataFrame to a CSV file
        df.to_csv(csv_file, index=False)
        print("Prepared data written to", csv_file)

    def prepare_data(csv_file, label_col_index, cols_to_encode=[], write_to_csv=True):
        """
        Prepares the data by loading a CSV file, one-hot encoding non-numerical columns,
        and optionally writing the prepared data to a new CSV file.

        Parameters:
        - csv_file (str): The path of the CSV file to load.
        - label_col_index (int): The index of the label column.
        - cols_to_encode (list): The list of column indices to one-hot encode. Default is an empty list.
        - write_to_csv (bool): Whether to write the prepared data to a new CSV file. Default is True.

        Returns:
        - df (pandas.DataFrame): The prepared DataFrame.
        """

        # Load the CSV file into a pandas DataFrame
        df = pd.read_csv(csv_file)

        # One-hot-encode non-numerical columns
        df = DataPrep.one_hot_encode(df, cols_to_encode)

        # Move the label column to the end
        label_col = df.columns[label_col_index]
        df = df[[c for c in df if c not in [label_col]] + [label_col]]

        if write_to_csv:
            prepared_csv_file = csv_file.replace('.csv', '_prepared.csv')
            DataPrep.write_data(df, prepared_csv_file)
            return df, prepared_csv_file

        return df, "N/A"
