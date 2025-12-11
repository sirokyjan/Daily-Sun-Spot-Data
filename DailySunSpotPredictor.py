import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_text as text
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import kagglehub
import shutil

hparams = {
    # Task specific constants
    "TASK_PROP__NUM_CLASSES": 3,

    # Dataset preprocessing
    "TRAINING_SPLIT": 0.9,
    "BATCH_SIZE": 16,
    "SHUFFLE_BUFFER_SIZE": 1000,
    "SPLIT_TIME": 0.9,
    "WINDOW_SIZE": 20,

    # Model params
    "OPTIMIZER_TYPE": 'adam',
    "LOSS_FUNCTION": 'sparse_categorical_crossentropy',
    "CONV_FILTERS_1": 32,
    "CONV_KERNEL_1": 5,
    "CONV_UNITS_1": 16,
    "LSTM_UNITS_1": 64,
    "LSTM_UNITS_2": 64,
    "DENSE_UNITS_1": 30,
    "DENSE_UNITS_2": 10,
    "L2_REG_RATE": 0.006,
    "DROPOUT": 0.6,
    #"KERNEL_INITIALIZER": 'glorot_uniform', # Added weight initializer. #'he_normal'
    #"BIAS_INITIALIZER": 'zeros',         # Added bias initializer

    # Training
    "LEARNING_RATE": 0.0001,
    "EARLY_STOP_PATIENCE": 10,
    "REDUCE_LR_PATIENCE": 5,
    "REDUCE_LR_FACTOR": 0.5,
    "REDUCE_LR_MIN_LR": 0.00001,
    "EPOCHS": 100
}

def download_dataset():

    # --- Your initial download code ---
    path_to_download_folder = kagglehub.dataset_download("abhinand05/daily-sun-spot-data-1818-to-2019")
    # --- End of initial download code ---

    print(f"\n--- Debugging: Contents of the Download Folder ---")
    print(f"Directory path: {path_to_download_folder}")
    print(f"Files found: {os.listdir(path_to_download_folder)}")
    print("---------------------------------------------------\n")

    file_name = "sunspot_data.csv"
    source_file_path = os.path.join(path_to_download_folder, file_name)


    destination_folder = "dataset"

    os.makedirs(destination_folder, exist_ok=True)
    destination_file_path = os.path.join(destination_folder, file_name)


    try:
        shutil.copy2(source_file_path, destination_file_path)
        print(f"✅ Success! File successfully copied from:")
        print(f"   Source: {source_file_path}")
        print(f"   Destination: {destination_file_path}")
        
        df = pd.read_csv(destination_file_path)
        print(f"\n✅ Data loaded successfully. Shape: {df.shape}")

    except FileNotFoundError:
        print(f"\n❌ Final Error: Could not find the file.")
        print(f"   Check that the file '{file_name}' is in the folder: {path_to_download_folder}")

    return df

def convert_missing_observations_format(df):
    # Replace the sentinel value (-1) with NaN
    df['Number of Sunspots'] = df['Number of Sunspots'].replace(-1, np.nan)

    # Optional: Check how many missing values you have now
    print(f"Total NaN values after conversion: {df['Number of Sunspots'].isna().sum()}")

def weighted_mean_triangular(data):
    """
    Calculates the Triangular Weighted Mean for a given window of data.
    Weights are highest in the center of the data window, prioritizing points
    closest to the imputation target.
    """
    # 1. Filter out NaNs
    data_cleaned = data.dropna()
    n = len(data_cleaned)
    
    if n == 0:
        return np.nan
    
    # 2. Create Triangular Weights
    # If n is 5, weights are [1, 2, 3, 2, 1]
    # If n is 6, weights are [1, 2, 3, 3, 2, 1] (or [1, 2, 3, 4, 3, 2] depending on definition, 
    # but the simple symmetric one is best here)
    
    # Determine the midpoint and create weights
    mid = n // 2
    
    # Weights for the first half (e.g., [1, 2])
    weights_first_half = np.arange(1, mid + 1)
    
    # Weights for the second half (in reverse order, e.g., [2, 1] for n=5)
    weights_second_half = np.arange(mid, 0, -1)
    
    if n % 2 != 0:
        # Odd length (e.g., n=5). Midpoint is unique and highest (e.g., [1, 2, 3] + [2, 1])
        weights = np.concatenate([weights_first_half, [mid + 1], weights_second_half])
    else:
        # Even length (e.g., n=6). Two middle points share the high weight.
        # Simple symmetric weights: [1, 2, 3, 3, 2, 1]
        weights = np.concatenate([weights_first_half, weights_first_half[::-1]])
        
    # Check that weights array matches the length of the cleaned data
    if len(weights) != n:
        # This shouldn't happen with the current logic, but a safeguard is useful
        # Let's adjust for the simplest symmetric case
        weights = np.concatenate([np.arange(1, mid + 1), np.arange(mid, 0, -1)])
        if len(weights) != n:
             # Fall back to linear if complexity is too high, but let's stick to the triangular goal
             pass # The previous logic should handle most practical cases for centered windows
             
    # 3. Calculate Weighted Average
    weighted_sum = np.sum(data_cleaned * weights)
    sum_of_weights = np.sum(weights)
    
    return weighted_sum / sum_of_weights

def calculate_rolling_exponential_average(data_series, span_size, adjust_mode=False):
    """
    Calculates the Exponential Weighted Moving Average (EWMA) for a pandas Series.
    The most recent/closest values have the highest (exponentially decaying) weights.

    Parameters:
    - data_series (pd.Series): The time series data (with -1s converted to NaNs).
    - span_size (int): The span (similar to a window size) used to calculate the decay factor alpha.
    - adjust_mode (bool): Whether to use the standard pandas adjustment formula. 
                          False (default) is typically preferred for imputation.
    
    Returns:
    - pd.Series: A new Series containing the calculated EWMA, rounded to the nearest integer.
    """
    
    # The .ewm() method handles NaNs automatically by skipping them and maintaining
    # the weighting decay relative to the time step (not the position in the window).
    rolling_ema = data_series.ewm(
        span=span_size,
        adjust=adjust_mode,
        min_periods=1  # We use min_periods=1 to fill data quickly
    ).mean()
    
    # Round the final calculated average to the nearest integer
    return rolling_ema.round(0)

def fill_missing_datapoints(df, target_column='Number of Sunspots'):
    """
    1. Converts -1 to NaN.
    2. Calculates a central Exponential Moving Average (CEMA) for imputation.
    3. Imputes the NaN values, then converts the result to the nullable integer type (Int64).
    """
    SPAN_SIZE=5

    # 1. Convert Sentinel Value (-1) to Standard Missing Value (NaN)
    df[target_column] = df[target_column].replace(-1, np.nan)
    
    # --- 2. Calculate the Central Exponential Moving Average (CEMA) ---
    
    # Forward EMA
    ema_forward = df[target_column].ewm(span=SPAN_SIZE, adjust=False, min_periods=1).mean()
    
    # Backward EMA
    ema_backward = df[target_column][::-1].ewm(span=SPAN_SIZE, adjust=False, min_periods=1).mean()[::-1]
    
    # Central EMA (The imputation values)
    df['Sunspots_CEMA'] = (ema_forward + ema_backward) / 2
    
    # 3. Imputation and Final Type Conversion
    
    # a) Impute using CEMA and round to the nearest whole number.
    imputed_series = df[target_column].fillna(df['Sunspots_CEMA']).round(0)
    
    # b) Use .astype('Int64') to allow NaNs to exist in the integer column.
    df['Number of Sunspots filled'] = imputed_series.astype('Int64') 
    
    # 4. Final verification and cleanup
    nan_count_before = df[target_column].isna().sum()
    # Check the final column for any remaining NaNs (should be very few, only for large initial/final gaps)
    nan_count_after = df['Number of Sunspots filled'].isna().sum()
    
    print("\n--- Imputation Summary (Central Exponential Average Fill) ---")
    print(f"Original NaNs in '{target_column}': {nan_count_before}")
    print(f"Remaining NaNs in 'Number of Sunspots filled': {nan_count_after} (Allowed due to Int64 type)")
    print(f"Data type of filled column: {df['Number of Sunspots filled'].dtype}")
    
    print("\n--- Data Check (CEMA Imputation) ---")
    print(df[[target_column, 'Sunspots_CEMA', 'Number of Sunspots filled']].head(30))
    
    return df

def save_all_data_to_csv_in_folder(df, folder_name="dataset", filename="Sunspots_Processed_WMA.csv", index_name='Date'):
    """
    Saves the entire DataFrame to a CSV file inside a specified folder.
    
    Args:
        df (pd.DataFrame): The DataFrame containing all data columns.
        folder_name (str): The name of the target folder (e.g., 'dataset').
        filename (str): The name of the output CSV file.
        index_name (str): The name to use for the date/time index column in the CSV.
    
    Returns:
        str: The full path to the saved file, or None on failure.
    """
    
    # 1. Create the full file path: 'dataset/Sunspots_Processed_WMA.csv'
    full_file_path = os.path.join(folder_name, filename)
    
    # 2. Check if the folder exists, and create it if it doesn't
    # 'exist_ok=True' prevents an error if the folder is already there
    os.makedirs(folder_name, exist_ok=True)
    
    # Ensure the index has a name for the CSV header
    if df.index.name is None:
        df.index.name = index_name
    
    # 3. Save the entire DataFrame to the constructed path
    try:
        df.to_csv(full_file_path, index=True)
        
        # Get the absolute path for confirmation
        absolute_path = os.path.abspath(full_file_path)
        
        print(f"\n✅ All data successfully saved!")
        print(f"Folder used: '{folder_name}'")
        print(f"File saved to: {absolute_path}")
        print(f"Total columns saved: {len(df.columns) + 1} (including the date index)")
        
        return absolute_path
    
    except Exception as e:
        print(f"\n❌ Error during file saving: {e}")
        return None

def print_data_head(*data, num_rows=5, column_name=None):
    """
    Prints the beginning of the DataFrame(s) and/or Series, with 
    the option to display a specific column.

    Args:
        *data: The variable(s) returned by the loading function (DataFrame/Series).
        num_rows (int): The number of rows to display.
        column_name (str, optional): The name of a specific column to display.
                                     If None, the entire DataFrame/Series head is printed.
    """
    
    def display_data(name, df_or_series):
        """Helper function to handle printing logic."""
        print(f"\n--- Displaying first {num_rows} rows of {name} ---")
        
        if column_name is not None and isinstance(df_or_series, pd.DataFrame):
            try:
                # 1. Select only the requested column
                data_to_print = df_or_series[[column_name]]
                print(f"(Column: '{column_name}')")
            except KeyError:
                print(f"❌ Error: Column '{column_name}' not found in {name}.")
                # Fallback to printing the full head if the column is not found
                data_to_print = df_or_series
        else:
            # 2. Print the full DataFrame/Series head
            data_to_print = df_or_series
            
        print(data_to_print.head(num_rows))


    if len(data) == 2:
        # Case 1: Features and Target
        features, target = data
        display_data("features", features)
        display_data("target", target)
        
    elif len(data) == 1:
        # Case 2: Single DataFrame/Series
        features = data[0]
        display_data("data", features)
        
    else:
        # Case 3: Invalid arguments
        print("Invalid number of arguments passed to print_data_head.")
        
    print("-" * 50)

def replace_yearly_data(sales_df, start_date_str, length_days, source_year):
    """
    Replaces a segment of a time series with data from the same period in a different year.
    This version is optimized for performance on large DataFrames.

    Args:
        sales_df (pd.DataFrame): DataFrame with sales data, including a 'date' column.
        start_date_str (str): The start date of the period to replace (e.g., "2023-01-15").
        length_days (int): The number of days to replace from the start date.
        source_year (int): The year to copy the data from.

    Returns:
        pd.DataFrame: A new DataFrame with the specified date range replaced.
    """
    
    df = sales_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # Define the target period to be replaced
    start_date = pd.to_datetime(start_date_str)
    end_date = start_date + pd.Timedelta(days=length_days - 1)
    
    # Define the source period to copy from
    source_start_date = start_date.replace(year=source_year)
    source_end_date = end_date.replace(year=source_year)
    
    # 1. Select the source data using a boolean mask
    source_mask = (df['date'] >= source_start_date) & (df['date'] <= source_end_date)
    source_data = df[source_mask].copy()

    # Check if the source data is empty
    if source_data.empty:
        print(f"Error: No data found for the source period between {source_start_date.date()} and {source_end_date.date()}. No changes were made.")
        return sales_df

    # 2. Calculate the time difference and shift the dates in the source data (fast vectorized operation)
    time_delta = start_date - source_start_date
    source_data['date'] = source_data['date'] + time_delta

    # 3. Create a mask for the target period to be removed from the original dataframe
    target_mask = (df['date'] >= start_date) & (df['date'] <= end_date)

    # 4. Combine the original data (excluding the target period) with the new source data
    result_df = pd.concat([df[~target_mask], source_data], ignore_index=True)

    # 5. Sort the final dataframe by date to restore the correct order
    result_df.sort_values(by='date', inplace=True)

    print(f"Successfully replaced data from {start_date.date()} to {end_date.date()} with data from {source_year}.")
    
    return result_df

def train_val_split(time, series):
    """ Splits time series into train and validations sets"""
    split_coord = int(len(time)*hparams['SPLIT_TIME'])

    time_train = time[:split_coord]
    series_train = series[:split_coord]
    time_valid = time[split_coord:]
    series_valid = series[split_coord:]

    return time_train, series_train, time_valid, series_valid

def generate_sales_dashboard(sales_df, events_df=None, output_filename="sales_dashboard.html"):
    """
    Generates an interactive HTML dashboard with sales data and optional event markers.

    Args:
        sales_df (pd.DataFrame): DataFrame with sales data. 
                                    Must contain 'date', 'family', and 'sales' columns.
        events_df (pd.DataFrame, optional): DataFrame with event data. 
                                            Must contain 'date' and 'description' columns. Defaults to None.
        output_filename (str): The name of the output HTML file.
    """
    
    # Prepare the Data 
    
    # Ensure date column is in datetime format
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    
    # Aggregate total sales per day for the first chart
    total_sales_by_date = sales_df.groupby('date')['sales'].sum().reset_index()
    
    # Pivot the data for the second chart to have one column per product family
    family_sales_by_date = sales_df.pivot_table(
        index='date', 
        columns='family', 
        values='sales', 
        aggfunc='sum'
    ).fillna(0)

    # Create the Figure with Subplots
    fig = make_subplots(
        rows=2, 
        cols=1, 
        subplot_titles=("Total Store Sales", "Sales by Product Family"),
        vertical_spacing=0.15
    )

    # Add Traces for the First Chart (Total Sales)
    fig.add_trace(
        go.Scatter(
            x=total_sales_by_date['date'], 
            y=total_sales_by_date['sales'], 
            mode='lines', 
            name='Total Sales',
            line=dict(color='royalblue')
        ), 
        row=1, col=1
    )

    # Add Traces for the Second Chart (Sales by Family)
    for family in family_sales_by_date.columns:
        fig.add_trace(
            go.Scatter(
                x=family_sales_by_date.index, 
                y=family_sales_by_date[family], 
                mode='lines', 
                name=family
            ), 
            row=2, col=1
        )
        
    # Add Event Markers if events_df is provided
    if events_df is not None:
        events_df['date'] = pd.to_datetime(events_df['date'])
        
        # Add an invisible scatter trace for hover text
        if not total_sales_by_date.empty:
            fig.add_trace(
                go.Scatter(
                    x=events_df['date'],
                    # Position markers near the top of the chart
                    y=[total_sales_by_date['sales'].max() * 0.95] * len(events_df),
                    mode='markers',
                    marker=dict(color='rgba(0,0,0,0)', size=10), # Invisible markers
                    hoverinfo='text',
                    text=events_df['description'], # Text to show on hover
                    name='Events',
                    showlegend=False
                ),
                row=1, col=1
            )

        for index, event in events_df.iterrows():
            # Add a vertical line for the event date on both charts
            fig.add_vline(x=event['date'], line_width=1, line_dash="dash", line_color="red", row="all", col=1)

    #Update Layout and Save
    fig.update_layout(
        title_text="Store Sales In Ecuador",
        height=800,
        legend_title_text='Product Family'
    )
    
    fig.update_yaxes(title_text="Total Sales", row=1, col=1)
    fig.update_yaxes(title_text="Sales per Family", row=2, col=1)

    try:
        fig.write_html(output_filename)
        print(f"Interactive chart saved successfully to '{output_filename}'")
    except IOError as e:
        print(f"Error saving chart: {e}")

def group_sales_by_family(sales_df):
    """
    Groups a sales DataFrame by the 'family' column and returns a list 
    of DataFrames, one for each product family.

    Args:
        sales_df (pd.DataFrame): The input DataFrame containing sales data
                                 with a 'family' column.

    Returns:
        list: A list of pandas DataFrames, where each DataFrame contains 
              the data for a single product family.
    """
    
    # Use the .groupby() method to create a grouping object
    grouped = sales_df.groupby('family')
    
    # Use a list comprehension to create a list of DataFrames from the groups.
    # The 'group' variable in the loop is the actual DataFrame for each family.
    # The 'name' variable (which we ignore with '_') is the family name (e.g., 'AUTOMOTIVE').
    list_of_dfs = [group for _, group in grouped]
    
    print(f"Successfully grouped the data into {len(list_of_dfs)} separate DataFrames.")
    
    return list_of_dfs

def windowed_dataset(series, window_size):
    """Creates windowed dataset"""
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    dataset = dataset.shuffle(hparams['SHUFFLE_BUFFER_SIZE'])
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))
    dataset = dataset.batch(hparams['BATCH_SIZE']).prefetch(1)
    return dataset

def create_uncompiled_model_wo_antioverfitting():
    """Define uncompiled model

    Returns:
        tf.keras.Model: uncompiled model
    """
  
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(hparams['WINDOW_SIZE'],1)),
        tf.keras.layers.Conv1D(filters=hparams['CONV_FILTERS_1'], kernel_size=hparams['CONV_KERNEL_1'], strides=1,padding="causal", activation="relu"),
        tf.keras.layers.LSTM(hparams['LSTM_UNITS_1'], return_sequences=True),
        tf.keras.layers.LSTM(hparams['LSTM_UNITS_2']),
        tf.keras.layers.Dense(hparams['DENSE_UNITS_1'], activation="relu"), 
        tf.keras.layers.Dense(hparams['DENSE_UNITS_2'], activation="relu"),
        tf.keras.layers.Dense(1)  
        ])
    
    return model

def adjust_learning_rate(dataset):
    """Fit model using different learning rates

    Args:
        dataset (tf.data.Dataset): train dataset

    Returns:
        tf.keras.callbacks.History: callback history
    """

    model = create_uncompiled_model_wo_antioverfitting()
    
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-5 * 10**(epoch / 20))
      
    # Select your optimizer
    optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    
    # Compile the model passing in the appropriate loss
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer, 
                  metrics=["mae"]) 

    history = model.fit(dataset, epochs=100, callbacks=[lr_schedule])
    
    return history

def create_and_compile_model(dense_layers, dense_units, dense_units_2, learning_rate, l2_rate, dropout_param):
    """
    Builds an advanced NLI model using pre-computed BERT embeddings
    and a cross-attention mechanism.
    """
    # --- 1. Define the two input layers for the pre-computed embeddings ---
    # The shape corresponds to (sequence_length, embedding_dim) from BERT
    input_premise_embedding = tf.keras.layers.Input(shape=(128, 768), name='input_premise_embedding')
    input_hypothesis_embedding = tf.keras.layers.Input(shape=(128, 768), name='input_hypothesis_embedding')

    # --- 2. Apply Cross-Attention ---
    # The hypothesis attends to the premise to create a context-aware representation
    attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
    attended_hypothesis = attention_layer(query=input_hypothesis_embedding, value=input_premise_embedding, key=input_premise_embedding)

    # --- 3. Pool the outputs to get sentence-level vectors (Symmetrical Pooling) ---
    premise_pooled = tf.keras.layers.GlobalAveragePooling1D()(input_premise_embedding)
    hypothesis_pooled = tf.keras.layers.GlobalAveragePooling1D()(attended_hypothesis)

    # --- 4. Concatenate for the Final Classifier ---
    concatenated = tf.keras.layers.concatenate(
        [premise_pooled, hypothesis_pooled], name='concatenated_layer'
    )

    # --- 5. Add the Classifier (Dense Layers) ---
    dense_1 = tf.keras.layers.Dense(dense_units, 
                                     activation='relu', 
                                     #kernel_initializer=kernel_initializer,
                                     #bias_initializer=bias_initializer,
                                     name='dense_1', 
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(concatenated)
    dropout_1 = tf.keras.layers.Dropout(dropout_param, name='dropout_1')(dense_1)

    if dense_layers>1:
        dense_2 = tf.keras.layers.Dense(dense_units_2, 
                                     activation='relu',
                                     #kernel_initializer=kernel_initializer,
                                     #bias_initializer=bias_initializer,
                                     name='dense_2', 
                                     kernel_regularizer=tf.keras.regularizers.l2(l2_rate))(dropout_1)
        dropout_2 = tf.keras.layers.Dropout(dropout_param, name='dropout_2')(dense_2)
    
    # --- 6. Define the Final Output Layer ---
    output = tf.keras.layers.Dense(hparams['TASK_PROP__NUM_CLASSES'], activation='softmax', name='output')(dropout_2 if dense_layers>1 else dropout_1)

    # --- 7. Build and Compile the Final Model ---
    model = tf.keras.Model(inputs=[input_premise_embedding, input_hypothesis_embedding], outputs=output)
 
    if hparams['OPTIMIZER_TYPE'] == 'adam':
        optimizer =tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                        #beta_1=0.9,
                                        #beta_2=0.999,
                                        #epsilon=1e-07,
                                        #amsgrad=False,
                                        #weight_decay=None,
                                        #clipnorm=None,
                                        #clipvalue=None,
                                        #global_clipnorm=None,
                                        #use_ema=False,
                                        #ema_momentum=0.99,
                                        #ema_overwrite_frequency=None,
                                        #loss_scale_factor=None,
                                        #gradient_accumulation_steps=None,
                                        name='adam',
                                        #**kwargs
                                    ) 
        
    elif hparams['OPTIMIZER_TYPE'] == 'rmsprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    
    else:
        print(f"Warning: Optimizer type '{hparams['OPTIMIZER_TYPE']}' not recognized. Defaulting to Adam.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer,
                  loss=hparams['LOSS_FUNCTION'],
                  #loss_weights=None,
                  metrics=['accuracy'],
                  #weighted_metrics=None,
                  #run_eagerly=False,
                  #steps_per_execution=1,
                  #jit_compile='auto',
                  #auto_scale_loss=True
                  )

    return model 
    

def plot_training_charts(training_history, json_log_path, output_filename):
    """
    Saves the Keras training history to JSON and plots the loss and
    accuracy metrics as an interactive HTML chart. This version displays
    only curves and includes the best values in the legend.

    Args:
        training_history: A Keras History object from model.fit().
        json_log_path (str): Path to save the history log.
        output_filename (str): Path to save the interactive HTML chart.
    """
    history_dict = training_history.history
    serializable_history = {key: [float(value) for value in values] for key, values in history_dict.items()}

    # --- 1. Save the history dictionary to a JSON file ---
    print(f"Saving training history log to {json_log_path}...")
    try:
        with open(json_log_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)
        print("History log saved successfully.")
    except IOError as e:
        print(f"Error saving history log: {e}")

    # --- 2. Create the interactive plot ---
    print(f"\nVisualizing training history and saving to {output_filename}...")
    
    if not history_dict or 'loss' not in history_dict or 'accuracy' not in history_dict:
        print("Warning: History is missing 'loss' or 'accuracy' keys. Cannot plot chart.")
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Loss', 'Model Accuracy'))
    epochs = list(range(1, len(history_dict['loss']) + 1))

    # --- Plot Loss (Training vs. Validation) ---
    min_loss = min(history_dict['loss'])
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=history_dict['loss'], 
        name=f'Training Loss (Min: {min_loss:.4f})', 
        mode='lines'  # Changed from 'lines+markers' to 'lines'
    ), row=1, col=1)
    
    if 'val_loss' in history_dict:
        min_val_loss = min(history_dict['val_loss'])
        fig.add_trace(go.Scatter(
            x=epochs, 
            y=history_dict['val_loss'], 
            name=f'Validation Loss (Min: {min_val_loss:.4f})', 
            mode='lines'
        ), row=1, col=1)

    # --- Plot Accuracy (Training vs. Validation) ---
    max_accuracy = max(history_dict['accuracy'])
    fig.add_trace(go.Scatter(
        x=epochs, 
        y=history_dict['accuracy'], 
        name=f'Training Accuracy (Max: {max_accuracy:.4f})', 
        mode='lines'
    ), row=1, col=2)
    
    if 'val_accuracy' in history_dict:
        max_val_accuracy = max(history_dict['val_accuracy'])
        fig.add_trace(go.Scatter(
            x=epochs, 
            y=history_dict['val_accuracy'], 
            name=f'Validation Accuracy (Max: {max_val_accuracy:.4f})', 
            mode='lines'
        ), row=1, col=2)

    # --- Update layout, titles, and legend ---
    fig.update_layout(
        title_text="Model Training History",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    # --- 3. Save the figure to an HTML file ---
    try:
        fig.write_html(output_filename)
        print(f"Interactive chart saved successfully to {output_filename}")
    except IOError as e:
        print(f"Error saving chart: {e}")

def save_predictions_to_csv(predictions, ids, output_path):
    """
    Saves the model's predictions to a CSV file in the required submission format.

    Args:
        predictions (np.ndarray): The output from the model.predict() method.
        ids (np.ndarray): The array of ids corresponding to the predictions.
        output_path (str): The file path where the submission CSV will be saved.
    """
    # Get the predicted label for each prediction
    predicted_labels = np.argmax(predictions, axis=1)

    # Create a pandas DataFrame with the specified column names
    submission_df = pd.DataFrame({
        'id': ids,
        'prediction': predicted_labels
    })

    # Save the DataFrame to a CSV file
    try:
        submission_df.to_csv(output_path, index=False)
        print(f"Successfully saved {len(predicted_labels)} predictions to {output_path}")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == '__main__':

    try:
        dataset_csv = download_dataset()
        print(dataset_csv)

        fill_missing_datapoints(dataset_csv)

        #print_data_head(dataset_csv, num_rows=20, column_name='Sunspots_WMA')

        save_all_data_to_csv_in_folder(dataset_csv, filename="Sunspots_Processed_WMA.csv", index_name='Date')

        exit()
        
        # Read the data from CSV files once
        data_train, _ = read_data_from_csv(TRAIN_DATA_CSV_PATH, is_training_data=True)
        features_test = read_data_from_csv(TEST_DATA_CSV_PATH, is_training_data=False)

        print_data_head(data_train)

        # Print the train data
        generate_sales_dashboard(data_train,  output_filename="ecuador_sales_dashboard.html")

        # A magnitude 7.8 earthquake struck Ecuador on April 16, 2016. People rallied in relief efforts donating water and other first need products which greatly affected supermarket sales for several weeks after the earthquake
        data_train_wo_earthquake = replace_yearly_data(
            sales_df=data_train, 
            start_date_str="2016-04-16", 
            length_days=30, 
            source_year=2017
        )
        events_data = [
            {'date': '2016-04-16', 'description': 'earthquake struck'},
        ]
        events_df = pd.DataFrame(events_data)
        generate_sales_dashboard(data_train_wo_earthquake, events_df=events_df,  output_filename="ecuador_sales_dashboard_wo_earthquake.html")

        families = data_train_wo_earthquake['family'].unique()
        print(families)

        grouped_by_family = group_sales_by_family(data_train_wo_earthquake)
        print(grouped_by_family[0].head(20))

        exit()

        #drop header
        data_train_wo_earthquake = data_train_wo_earthquake[1:]

        time_train, features_train, time_valid, features_valid = train_val_split(data_train_wo_earthquake['date'], data_train_wo_earthquake['sales'])

        train_dataset = windowed_dataset(features_train, window_size=hparams['WINDOW_SIZE'])

        #learning rate optimization
        lr_history = adjust_learning_rate(train_dataset)
        plt.semilogx(lr_history.history["learning_rate"], lr_history.history["loss"])

        uncompiled_model = create_uncompiled_model_wo_antioverfitting()
        uncompiled_model.summary()
    
        print(f"Training dataset contains {len(train_dataset_csv)} examples\n")
        print(f"Test dataset contains {len(test_dataset_raw)} examples\n")

        SHUFFLE_BUFFER_SIZE = 1000
        PREFETCH_BUFFER_SIZE = tf.data.AUTOTUNE
        train_dataset_final = train_dataset.cache().shuffle(SHUFFLE_BUFFER_SIZE).prefetch(PREFETCH_BUFFER_SIZE).batch(hparams['BATCH_SIZE'])
        validation_dataset_final = validation_dataset.cache().prefetch(PREFETCH_BUFFER_SIZE).batch(hparams['BATCH_SIZE'])

        print(f"Buffered {SHUFFLE_BUFFER_SIZE} elements for the training dataset.")

        print()

        print(f"There are {len(train_dataset_final)} batches for a total of {hparams['BATCH_SIZE']*len(train_dataset_final)} elements for training.\n")
        print(f"There are {len(validation_dataset_final)} batches for a total of {hparams['BATCH_SIZE']*len(validation_dataset_final)} elements for validation.\n")

        print()

        print(f"Create and compile model")
        nn_model = create_and_compile_model(
            hparams['DENSE_LAYERS'], 
            hparams['DENSE_UNITS_1'],
            hparams['DENSE_UNITS_2'], 
            hparams['LEARNING_RATE'],
            hparams['L2_REG_RATE'],
            hparams['DROPOUT']
        )
        nn_model.summary()

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',    # Monitor the validation loss
            #min_delta = 0,
            patience=hparams['EARLY_STOP_PATIENCE'],            # Stop if val_loss doesn't improve for 5 epochs
            verbose=1,
            #mode='auto',
            #baseline=None,
            restore_best_weights=True # Restore model weights from the epoch with the best val_loss
            #start_from_epoch=0
        )

        learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(   monitor='val_accuracy',
                                                factor=hparams['REDUCE_LR_FACTOR'],
                                                patience=hparams['REDUCE_LR_PATIENCE'],
                                                verbose=1,
                                                #mode='auto',
                                                #min_delta=0.0001,
                                                #cooldown=0,
                                                min_lr=hparams['REDUCE_LR_MIN_LR'],
                                                #**kwargs
                                            )

        training_history = nn_model.fit(x=train_dataset_final,
                                        #y=None,
                                        #batch_size=None,
                                        epochs=hparams['EPOCHS'],
                                        #verbose='auto',
                                        callbacks=[early_stopping_callback, learning_rate_scheduler],
                                        #validation_split=0.0,
                                        validation_data=validation_dataset_final,
                                        #shuffle=True,
                                        #class_weight=None,
                                        #sample_weight=None,
                                        #initial_epoch=0,
                                        #steps_per_epoch=None,
                                        #validation_steps=None,
                                        #validation_batch_size=None,
                                        #validation_freq=1
                                        ) 
        
        print()

        history_log_path = os.path.join(current_dir, "training_history.json")
        loss_acc_chart_path = os.path.join(current_dir, "training_loss_acc_chart.html")
        plot_training_charts(training_history, history_log_path, loss_acc_chart_path)

        SAVED_MODEL_PATH = os.path.join(current_dir, "trained_model_complete.tf")
        nn_model.save(SAVED_MODEL_PATH, save_format='tf')
        print("\nModel saved to {SAVED_MODEL_PATH}")

        predictions = nn_model.predict(test_dataset_raw.batch(hparams['BATCH_SIZE']), #x
                                       #batch_size=None, 
                                       verbose=False, 
                                       #steps=None, 
                                       #callbacks=None
                                       )

        prediction_csv_path = os.path.join(current_dir, "submission.csv")
        save_predictions_to_csv(predictions, ids_test, prediction_csv_path)

        # Print the hyperparameter table
        print("-" * 50)
        print("Hyperparameters for this run:")
        for key, value in hparams.items():
            print(f"{key:<25}: {value}")
        print("-" * 50)

    except ValueError as e:
        print(f"\nError: {e}")