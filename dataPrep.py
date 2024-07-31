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
        for col in cols:                                # For each column index
            unique_values = df.iloc[:, col].unique()    # Get the unique values in the column

            for value in unique_values:                                 # For each unique value, create a new binary column
                df[str(value)] = (df.iloc[:, col] == value).astype(int) # Set the value to 1 if the original column is equal to the unique value

        df = df.drop(df.columns[cols], axis=1)          # Drop the original column

        return df

    def write_data(df, csv_file):
        """
        Writes the DataFrame to a CSV file.

        Parameters:
        - df (pandas.DataFrame): The DataFrame to be written.
        - csv_file (str): The path of the CSV file to write to.
        """
        df.to_csv(csv_file, index=False)            # Write the DataFrame to a CSV file
        print("Prepared data written to", csv_file) # Print the path of the written file

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
        df = pd.read_csv(csv_file)              # Load the CSV file
        
        label_col = df.columns[label_col_index]             # Get the label column name(s)
        df = DataPrep.one_hot_encode(df, cols_to_encode)    # One-hot encode the specified columns

        df = pd.concat([df.drop(label_col, axis=1), df[[label_col]]], axis=1)   # Move the label column from start to the end

        if write_to_csv:    # If write_to_csv is True
            prepared_csv_file = csv_file.replace('.csv', '_prepared.csv')   # Create a new file path for the prepared data
            DataPrep.write_data(df, prepared_csv_file)                      # Write the prepared data to a new CSV file
            return df, prepared_csv_file                                    # Return the prepared DataFrame and the path of the written file

        return df, "N/A"    # Else, return the prepared DataFrame and "N/A"
        