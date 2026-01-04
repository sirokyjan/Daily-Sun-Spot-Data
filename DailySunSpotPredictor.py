import loss_functions as MyLossLib
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import kagglehub
import shutil
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LR_FINDER = 0
LR_FINDER_TEST = 0
LOSS_FNC_TEST = 1

hparams = {
    # Dataset preprocessing
    #"TRAINING_SPLIT": 0.9,
    "BATCH_SIZE": 1024,
    "SHUFFLE_BUFFER_SIZE": 20000,
    "SPLIT_TIME": 0.9,
    "WINDOW_SIZE": 50,

    # Model params
    "OPTIMIZER_TYPE": 'adam',
    "LOSS_FUNCTION": tf.keras.losses.Huber(),
    "CONV_FILTERS_1": 32,
    "CONV_KERNEL_1": 9,
    "LSTM_UNITS_1": 64,
    "LSTM_UNITS_2": 64,
    "DENSE_UNITS_1": 30,
    "DENSE_UNITS_2": 10,
    "L2_REG_RATE": 0.006,
    "DROPOUT": 0.2,
    #"KERNEL_INITIALIZER": 'glorot_uniform', # Added weight initializer. #'he_normal'
    #"BIAS_INITIALIZER": 'zeros',         # Added bias initializer

    # Learning rate finder
    "LRF_NUM_EPOCHS": 60,
    "LRF_START_LR": 1e-7,
    "LRF_GROWTH_DENOMINATOR": 10,

    # Training
    "LEARNING_RATE": 1.6e-06,
    "EARLY_STOP_PATIENCE": 10,
    "REDUCE_LR_PATIENCE": 8,
    "REDUCE_LR_FACTOR": 0.5,
    "REDUCE_LR_MIN_LR": 1.6e-06/32,
    "EPOCHS": 800
}

def download_dataset():

    path_to_download_folder = kagglehub.dataset_download("abhinand05/daily-sun-spot-data-1818-to-2019")

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

def fill_missing_with_linear_extrapolation(df, target_column='Number of Sunspots'):
    """
    1. Converts -1 to NaN.
    2. Fills internal gaps using linear interpolation.
    3. Fills leading/trailing gaps using linear extrapolation based on the trend.
    """
    # 1. Replace sentinel -1 with NaN
    df[target_column] = df[target_column].replace(-1, np.nan)
    
    # 2. Linear Interpolation (Internal Gaps)
    # This connects two known points with a straight line.
    df['Sunspots_Linear'] = df[target_column].interpolate(method='linear')
    
    # 3. Linear Extrapolation (Edges)
    # 'limit_direction="both"' fills NaNs at the start and end of the series.
    # 'fill_value="extrapolate"' uses the slope of the nearest points to project outward.
    df['Sunspots_Linear'] = df['Sunspots_Linear'].interpolate(
        method='linear', 
        limit_direction='both', 
        fill_value='extrapolate'
    )
    
    # 4. Cleanup: Round and ensure non-negative (sunspots can't be negative)
    df['Number of Sunspots filled'] = df['Sunspots_Linear'].round(0).clip(lower=0)
    
    # Statistics for verification
    nan_count = df['Number of Sunspots filled'].isna().sum()
    print(f"--- Linear Extrapolation Summary ---")
    print(f"Remaining NaNs: {nan_count}")
    
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
    
    # Create the full file path: 'dataset/Sunspots_Processed_WMA.csv'
    full_file_path = os.path.join(folder_name, filename)
    
    # Check if the folder exists, and create it if it doesn't
    # 'exist_ok=True' prevents an error if the folder is already there
    os.makedirs(folder_name, exist_ok=True)
    
    # Ensure the index has a name for the CSV header
    if df.index.name is None:
        df.index.name = index_name
    
    # Save the entire DataFrame to the constructed path
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
                # Select only the requested column
                data_to_print = df_or_series[[column_name]]
                print(f"(Column: '{column_name}')")
            except KeyError:
                print(f"❌ Error: Column '{column_name}' not found in {name}.")
                # Fallback to printing the full head if the column is not found
                data_to_print = df_or_series
        else:
            # Print the full DataFrame/Series head
            data_to_print = df_or_series
            
        print(data_to_print.head(num_rows))


    if len(data) == 2:
        features, target = data
        display_data("features", features)
        display_data("target", target)
        
    elif len(data) == 1:
        features = data[0]
        display_data("data", features)
        
    else:
        print("Invalid number of arguments passed to print_data_head.")
        
    print("-" * 50)

def delete_first_x_data_rows(df, rows_to_skip):
    """
    Deletes the first X data rows from a loaded pandas DataFrame, 
    preserving the header (column names).

    Args:
        df (pd.DataFrame): The input DataFrame that has already been loaded.
        rows_to_skip (int): The number of data rows (after the header) to delete.
                            Must be a non-negative integer.
    
    Returns:
        pd.DataFrame: A new DataFrame with the first X data rows removed.
                      Returns None if an error occurs.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input must be a pandas DataFrame.")
        return None
        
    if rows_to_skip < 0:
        print("Error: rows_to_skip must be a non-negative integer.")
        return None
        
    if rows_to_skip >= len(df):
        print(f"Warning: Attempted to skip {rows_to_skip} rows in a DataFrame with only {len(df)} rows (excluding header).")
        # Return an empty DataFrame with the original columns
        return pd.DataFrame(columns=df.columns)

    try:
        df_modified = df.iloc[rows_to_skip:]
        
        print(f"Successfully removed the first {rows_to_skip} data rows.")
        return df_modified

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def extract_column_to_numpy(df: pd.DataFrame, column_name: str) -> np.ndarray:
    """
    Extracts a specified column from a pandas DataFrame and converts it 
    into a NumPy array.

    Args:
        df (pd.DataFrame): The input DataFrame that has already been loaded.
        column_name (str): The name of the column to be extracted.
    
    Returns:
        np.ndarray: A 1-dimensional NumPy array containing the data 
                    from the specified column. Returns None if the column is not found.
    """
    if not isinstance(df, pd.DataFrame):
        print("Error: Input must be a pandas DataFrame.")
        return None

    if column_name not in df.columns:
        print(f"Error: Column '{column_name}' not found in the DataFrame.")
        return None
    
    try:
        # Select the column (returns a pandas Series)
        column_series = df[column_name]
        
        # Convert the pandas Series to a NumPy array
        # .to_numpy() is the modern and preferred method for conversion.
        numpy_array = column_series.to_numpy()
        
        print(f"Successfully extracted column '{column_name}' and converted it to a NumPy array.")
        print(f"Resulting array shape: {numpy_array.shape}")
        
        return numpy_array

    except Exception as e:
        print(f"An unexpected error occurred during conversion: {e}")
        return None

def cut_first_column_to_numpy(df: pd.DataFrame) -> np.ndarray:
    """
    Removes the first column from a pandas DataFrame and converts the 
    removed column into a 1D NumPy array.
    
    NOTE: This function modifies the input DataFrame in-place (cuts the column).

    Args:
        df (pd.DataFrame): The input DataFrame.
    
    Returns:
        np.ndarray: A 1-dimensional NumPy array of the data from the 
                    removed first column. Returns None if the DataFrame 
                    is empty or invalid.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        print("Error: Input is not a valid or non-empty pandas DataFrame.")
        return None

    try:
        # Get the column name for printing/checking
        first_column_name = df.columns[0]
        
        # Use .pop() to remove the column AND return it as a pandas Series
        # df.pop() is the method that performs the "cut" and modifies the df in-place.
        removed_series = df.pop(first_column_name)
        
        # Convert the pandas Series to a NumPy array
        numpy_array = removed_series.to_numpy()
        
        print(f"Successfully cut column '{first_column_name}' from DataFrame.")
        print(f"The modified DataFrame now has {df.shape[1]} columns.")
        
        return numpy_array

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

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


def windowed_dataset(series, window_size, shuffle=True):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    
    # 1. Windowing
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    
    # 2. Cache the flattened windows (Avoids re-computing windows every epoch)
    dataset = dataset.cache() 
    
    # 3. Shuffle (Done on the cached data)
    if shuffle:
        dataset = dataset.shuffle(hparams['SHUFFLE_BUFFER_SIZE'], reshuffle_each_iteration=True)
    
    # 4. Map (X, Y split)
    dataset = dataset.map(lambda window: (window[:-1], window[-1]), 
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    # 5. Batch and Prefetch (THE FINAL STEP)
    # This ensures the GPU always has a batch ready in local memory.
    dataset = dataset.batch(hparams['BATCH_SIZE']).prefetch(tf.data.AUTOTUNE)
    
    return dataset

def create_uncompiled_model():
    """Define uncompiled model

    Returns:
        tf.keras.Model: uncompiled model
    """
  
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=(hparams['WINDOW_SIZE'],1)),
        tf.keras.layers.Conv1D(filters=hparams['CONV_FILTERS_1'], kernel_size=hparams['CONV_KERNEL_1'], strides=1,padding="causal", activation="relu"),
        tf.keras.layers.LayerNormalization(), # Added for stability
        tf.keras.layers.LSTM(hparams['LSTM_UNITS_1'], return_sequences=True),
        tf.keras.layers.LSTM(hparams['LSTM_UNITS_2']),
        tf.keras.layers.Dense(hparams['DENSE_UNITS_1'], activation="relu"),
        #tf.keras.layers.Dropout(hparams['DROPOUT'], name='dropout_1'),
        tf.keras.layers.Dense(hparams['DENSE_UNITS_2'], activation="relu"),
        #tf.keras.layers.Dropout(hparams['DROPOUT'], name='dropout_2'),
        tf.keras.layers.Dense(1)  
        ])

    return model

def compile_model(model):

    if hparams['OPTIMIZER_TYPE'] == 'adam':
        optimizer =tf.keras.optimizers.Adam(learning_rate=hparams['LEARNING_RATE'],
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
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=hparams['LEARNING_RATE'])
        
    else:
        print(f"Warning: Optimizer type '{hparams['OPTIMIZER_TYPE']}' not recognized. Defaulting to Adam.")
        optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['LEARNING_RATE'])

    model.compile(optimizer=optimizer,
                  loss=hparams['LOSS_FUNCTION'],
                  #loss_weights=None,
                  metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')],
                  #weighted_metrics=None,
                  #run_eagerly=False,
                  #steps_per_execution=1,
                  #jit_compile='auto' #dont use XLA, its incompatible with LSTM
                  #auto_scale_loss=True
                  )

    return model

def plot_learningrate_loss_chart(x, y, xmin, xmax, ymin, ymax):

    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.tick_params('both', length=10, width=1, which='both')
    plt.semilogx(x, y)
    plt.xlabel("Learning Rate (Log Scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.show()  

def generate_interactive_lr_finder_chart(history, start_lr, growth_denominator, output_filename="lr_finder_chart.html"):
    """
    Generates a standalone interactive HTML chart from the LR Finder history.
    
    Args:
        history: The Keras History object from adjust_learning_rate.
        start_lr (float): The starting learning rate used in the schedule.
        growth_denominator (float): The denominator used in the LR growth formula.
        output_filename (str): The name of the HTML file to save.
    """
    # 1. Reconstruct Learning Rates and Extract Losses
    losses = np.array(history.history['loss'])
    epochs = np.arange(len(losses))
    lrs = start_lr * (10 ** (epochs / growth_denominator))

    # 2. Clean data (handle potential NaNs if the model diverged at high LRs)
    valid_mask = np.isfinite(losses)
    clean_lrs = lrs[valid_mask]
    clean_losses = losses[valid_mask]

    # 3. Create Plotly Figure
    fig = go.Figure()

    # Add the main loss curve
    fig.add_trace(go.Scatter(
        x=clean_lrs, 
        y=clean_losses,
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Learning Rate</b>: %{x:.2e}<br><b>Loss</b>: %{y:.4f}<extra></extra>'
    ))

    # 4. Professional Formatting
    fig.update_layout(
        title="Learning Rate Finder Analysis",
        xaxis_title="Learning Rate (Log Scale)",
        yaxis_title="Loss",
        xaxis_type="log", # Set X-axis to logarithmic
        template="plotly_white",
        hovermode="x unified",
        showlegend=True,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )

    # Add a grid for easier reading
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray', minor=dict(showgrid=True))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')

    # 5. Save and Notify
    try:
        fig.write_html(output_filename)
        print(f"✅ Interactive LR chart successfully saved to: {output_filename}")
    except Exception as e:
        print(f"❌ Failed to save LR chart: {e}")


def adjust_learning_rate(dataset):
    """
    Performs a Learning Rate Range Test on the model.

    This function trains the model for a short duration while exponentially 
    increasing the learning rate to find the optimal range.

    Args:
        dataset (tf.data.Dataset): The training dataset.

    Returns:
        tf.keras.callbacks.History: The training history containing loss vs. step data.
    """

    model = create_uncompiled_model()
    
    # Define exponential learning rate schedule
    lr_schedule = tf.keras.callbacks.LearningRateScheduler(
        lambda epoch: hparams['LRF_START_LR'] * 10**(epoch / hparams['LRF_GROWTH_DENOMINATOR'])
    )
      
    optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['LRF_START_LR'], global_clipnorm=1.0, epsilon=1e-5)
    
    # Compile model with Huber loss (robust choice)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer, 
                  metrics=["mae"]) 
    
    # Use a small, repeated dataset segment for quick testing
    mini_dataset = dataset.take(hparams['LRF_NUM_EPOCHS']).cache().repeat()

    # Run the test
    history = model.fit(mini_dataset, epochs=hparams['LRF_NUM_EPOCHS'], steps_per_epoch=60, callbacks=[lr_schedule])

    # Calculate LR and Loss values for plotting
    lrs = hparams['LRF_START_LR'] * (10 ** (np.arange(hparams['LRF_NUM_EPOCHS']) / hparams['LRF_GROWTH_DENOMINATOR']))  
    final_lr = hparams['LRF_START_LR'] * (10 ** (hparams['LRF_NUM_EPOCHS'] / hparams['LRF_GROWTH_DENOMINATOR'])) 
    loss_data = history.history["loss"]
    min_loss = np.min(loss_data)
    max_loss = np.max(loss_data) 

    # Plot the results
    generate_interactive_lr_finder_chart(
                history=history, 
                start_lr=hparams['LRF_START_LR'], 
                growth_denominator=hparams['LRF_GROWTH_DENOMINATOR'], 
                output_filename="LR_Finder.html"
            )
    smoothed_loss = pd.Series(history.history['loss']).ewm(span=10).mean()
    plot_learningrate_loss_chart(lrs, loss_data, hparams['LRF_START_LR'], final_lr, min_loss * 0.9, max_loss * 1.1)
    #plot_learningrate_loss_chart(lrs, smoothed_loss, hparams['LRF_START_LR'], final_lr, min_loss * 0.9, max_loss * 1.1)
    
    return history

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

    # Save the history dictionary to a JSON file
    print(f"Saving training history log to {json_log_path}...")
    try:
        with open(json_log_path, 'w') as f:
            json.dump(serializable_history, f, indent=4)
        print("History log saved successfully.")
    except IOError as e:
        print(f"Error saving history log: {e}")

    # Create the interactive plot
    print(f"\nVisualizing training history and saving to {output_filename}...")
    
    if not history_dict or 'loss' not in history_dict or 'accuracy' not in history_dict:
        print("Warning: History is missing 'loss' or 'accuracy' keys. Cannot plot chart.")
        return

    fig = make_subplots(rows=1, cols=2, subplot_titles=('Model Loss', 'Model Accuracy'))
    epochs = list(range(1, len(history_dict['loss']) + 1))

    # Plot Loss (Training vs. Validation)
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

    # Plot Accuracy (Training vs. Validation)
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

    # Update layout, titles, and legend
    fig.update_layout(
        title_text="Model Training History",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)

    # Save the figure to an HTML file
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
    predicted_labels = predictions.flatten()

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

def generate_timeseries_analysis_chart(time, series, title="Sunspot Analysis", output_filename="Sunspot dataset chart.html"):
    fig = go.Figure()

    # 1. Main Trace: Daily Data
    fig.add_trace(go.Scattergl(
        x=time,
        y=series,
        mode='lines',
        name='Daily Count',
        line=dict(color='#1f77b4', width=0.8),
        opacity=0.4 # Fade daily data slightly to make trends visible
    ))

    # 2. ADDED: 365-day Rolling Average (Optional but highly recommended)
    # This helps visualize the 11-year cycle through the daily noise
    df_temp = pd.DataFrame({'val': series}, index=pd.to_datetime(time))
    rolling_mean = df_temp['val'].rolling(window=365, center=True).mean()

    fig.add_trace(go.Scattergl(
        x=time,
        y=rolling_mean,
        mode='lines',
        name='1y Moving Avg',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Calendar Year",
        yaxis_title="Sunspot Number",
        hovermode="x unified",
        xaxis=dict(
            type="date", # This makes 'year' buttons work correctly
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(count=11, label="11y Cycle", step="year", stepmode="backward"),
                    dict(count=22, label="22y Cycle", step="year", stepmode="backward"),
                    dict(step="all", label="All Time")
                ])
            ),
            rangeslider=dict(visible=True)
        )
    )

    fig.write_html(output_filename)
    print(f"✅ Success: Chart saved. Daily points: {len(series)}")


def plot_training_histories(histories, filename="training_progress.html"):
    """
    Plots Training vs Validation MAE and RMSE from multiple histories 
    into a 2x2 interactive HTML grid.
    """
    fig = make_subplots(
        rows=2, cols=2, 
        subplot_titles=(
            'Training MAE', 'Training RMSE',
            'Validation MAE', 'Validation RMSE'
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )

    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']

    for i, history in enumerate(histories):
        h_dict = history.history if hasattr(history, 'history') else history
        
        model_label = getattr(history, 'user_defined_name', f'Model {i+1}')
        
        mae = h_dict.get('mae', [])
        rmse = h_dict.get('rmse', [])
        val_mae = h_dict.get('val_mae', [])
        val_rmse = h_dict.get('val_rmse', [])
        
        if mae:
            epochs = list(range(1, len(mae) + 1))
        else:
            continue # Skip if no data
            
        color = colors[i % len(colors)]
        
        def add_trace(data, name, row, col, show_legend=False):
            if data:
                fig.add_trace(
                    go.Scatter(
                        x=epochs, y=data,
                        mode='lines',
                        name=name,
                        line=dict(color=color),
                        legendgroup=f'group{i}',
                        showlegend=show_legend,
                        hovertemplate='<b>' + model_label + '</b><br>Epoch: %{x}<br>Value: %{y:.4f}'
                    ),
                    row=row, col=col
                )

        # Use the specific model name in the legend
        # We only show the legend for the first plot to avoid clutter
        add_trace(mae, model_label, 1, 1, show_legend=True)
        add_trace(rmse, f'{model_label} RMSE', 1, 2)
        add_trace(val_mae, f'{model_label} Val MAE', 2, 1)
        add_trace(val_rmse, f'{model_label} Val RMSE', 2, 2)

    fig.update_layout(
        title='Multi-Model Training vs Validation Performance',
        template='plotly_white',
        hovermode='x unified',
        height=800,
        legend_title='Models'
    )

    # Label Axes
    fig.update_xaxes(title_text='Epochs', row=2, col=1)
    fig.update_xaxes(title_text='Epochs', row=2, col=2)
    fig.update_yaxes(title_text='MAE', row=1, col=1)
    fig.update_yaxes(title_text='RMSE', row=1, col=2)
    fig.update_yaxes(title_text='Val MAE', row=2, col=1)
    fig.update_yaxes(title_text='Val RMSE', row=2, col=2)

    fig.write_html(filename)
    print(f"✅ Interactive comparison chart saved to {filename}")

def generate_model_predictions(model, series, window_size, batch_size, train_min, train_max):
    # Ensure series is a tensor with correct shape
    dataset = tf.data.Dataset.from_tensor_slices(series)
    
    # Use the cleaner batching method to avoid AutoGraph warnings
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size))
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    print("Generating predictions...")
    predictions = model.predict(dataset, verbose=0)
    
    # Inverse Scaling
    unscaled_predictions = predictions.flatten() * (train_max - train_min) + train_min
    
    # Post-processing: Sunspots are non-negative integers
    return np.maximum(0, np.round(unscaled_predictions))

def evaluate_predictions(actual, predictions, model_name):
    """
    Calculates various regression metrics to evaluate model performance.
    """
    # Mean Absolute Error (Average spot miss)
    mae = mean_absolute_error(actual, predictions)
    
    # Root Mean Squared Error (Penalizes large outliers/peaks)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    
    # R-Squared (Percentage of variance explained)
    r2 = r2_score(actual, predictions)
    
    # sMAPE (Symmetric Mean Absolute Percentage Error)
    # Good for time series with zeros/low values
    smape = 100/len(actual) * np.sum(2 * np.abs(predictions - actual) / (np.abs(actual) + np.abs(predictions) + 1e-10))

    return {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "sMAPE": smape
    }

def generate_comparison_dashboard(time, actual_series, predictions_list, metrics_list, output_filename="Prediction_Dashboard.html"):
    """
    Creates an interactive Plotly chart with a performance metrics table.
    Expects arrays that are ALREADY aligned in length.
    """
    # No internal window_offset slicing
    plot_time = time
    plot_actual = actual_series

    # Create Subplots: Row 1 is the Table, Row 2 is the Chart
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.15,
        specs=[[{"type": "table"}], [{"type": "scatter"}]],
        subplot_titles=("Model Performance Metrics", "Sunspot Forecast vs Actual")
    )

    # Add Metrics Table
    names = [m['Model'] for m in metrics_list]
    maes = [f"{m['MAE']:.2f}" for m in metrics_list]
    rmses = [f"{m['RMSE']:.2f}" for m in metrics_list]
    r2s = [f"{m['R2']:.3f}" for m in metrics_list]
    smapes = [f"{m['sMAPE']:.2f}%" for m in metrics_list]

    fig.add_trace(
        go.Table(
            header=dict(values=['<b>Model</b>', '<b>MAE</b>', '<b>RMSE</b>', '<b>R2 Score</b>', '<b>sMAPE</b>'],
                        fill_color='royalblue', font=dict(color='white', size=12), align='left'),
            cells=dict(values=[names, maes, rmses, r2s, smapes],
                       fill_color='lavender', align='left')
        ),
        row=1, col=1
    )

    # Add Actual Data Trace
    fig.add_trace(
        go.Scatter(x=plot_time, y=plot_actual, name='Actual', line=dict(color='black', width=2)),
        row=2, col=1
    )

    # Add Prediction Traces
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']
    for i, pred in enumerate(predictions_list):
        fig.add_trace(
            go.Scatter(x=plot_time, y=pred, name=names[i], 
                       line=dict(width=1.5, color=colors[i % len(colors)])),
            row=2, col=1
        )

    # Update Layout with Slider
    fig.update_layout(
        height=900,
        template="plotly_white",
        hovermode="x unified",
        xaxis2=dict(rangeslider=dict(visible=True), type="date")
    )

    fig.write_html(output_filename)
    print(f"✅ Full dashboard with metrics saved to: {output_filename}")

def analyze_lr_history(history, start_lr, growth_denominator=10):
    losses = np.array(history.history['loss'])
    lrs = start_lr * (10 ** (np.arange(len(losses)) / growth_denominator))

    valid_idx = np.isfinite(losses)
    losses, lrs = losses[valid_idx], lrs[valid_idx]

    # Find key indices
    min_idx = np.argmin(losses)
    # Use a gradient of the loss to find the steepest descent
    grads = np.gradient(losses)
    steepest_idx = np.argmin(grads)

    # Strategy A: One order of magnitude before the minimum
    safe_min_lr = lrs[min_idx] / 10
    
    # Strategy B: The steepest slope
    steepest_lr = lrs[steepest_idx]

    # Final Recommendation: Usually the smaller of the two for high stability
    recommended_lr = min(safe_min_lr, steepest_lr)

    print(f"Minimum Loss LR:   {lrs[min_idx]:.2e}")
    print(f"Steepest Slope LR: {steepest_lr:.2e}")
    print(f"Safe/Stable LR:    {recommended_lr:.2e}")
    
    return recommended_lr

if __name__ == '__main__':

    try:
        dataset_csv = download_dataset()
        print(dataset_csv)

        dataset_csv = fill_missing_with_linear_extrapolation(dataset_csv)

        #print_data_head(dataset_csv, num_rows=20, column_name='Sunspots_WMA')

        SKIP_ROWS = 7

        dataset_csv = delete_first_x_data_rows(dataset_csv, rows_to_skip=SKIP_ROWS)

        save_all_data_to_csv_in_folder(dataset_csv, filename="Sunspots_linear_extrapolation.csv", index_name='Date')

        sun_spot_timeseries = extract_column_to_numpy(dataset_csv, column_name = 'Number of Sunspots filled').astype('float32')
        time_line = cut_first_column_to_numpy(dataset_csv)

        print("sun_spot_timeseries")
        print(sun_spot_timeseries[0:5])
        print("time_timeseries")
        print(time_line[0:5])

        # Assuming your dataset starts on January 1st, 1818
        start_date = np.datetime64('1818-01-08') #(SKIP_ROWS)
        time_converted = start_date + np.array(time_line, dtype='timedelta64[D]')
        generate_timeseries_analysis_chart(time_line, sun_spot_timeseries, title="Sunspot Data Verification")

        print("train_val_split")
        time_train, features_train, time_valid, features_valid = train_val_split(time_line, sun_spot_timeseries)

        print("standardization")
        train_min, train_max = features_train.min(), features_train.max()
        features_train = (features_train - train_min) / (train_max - train_min)
        features_valid = (features_valid - train_min) / (train_max - train_min)
        features_train = features_train.astype('float32')
        features_valid = features_valid.astype('float32')

        print("windowed_dataset")
        train_dataset = windowed_dataset(features_train, window_size=hparams['WINDOW_SIZE'])
        validation_dataset = windowed_dataset(features_valid, window_size=hparams['WINDOW_SIZE'], shuffle = False)

        train_dataset_final = train_dataset
        validation_dataset_final = validation_dataset

        if LR_FINDER:
            print("learning rate optimization")
            lr_history = adjust_learning_rate(train_dataset)
            recomended_lr = analyze_lr_history(lr_history, hparams['LRF_START_LR'], hparams['LRF_GROWTH_DENOMINATOR'])
            print(f"Recomended Learning Rate used: {recomended_lr}")

            if LR_FINDER_TEST:
                # Define the learning rates you want to test
                learning_rates = [4e-3, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4]
                models = []
                histories = []

                print("\nLR Finder test: Create and compile models")

                for i, lr in enumerate(learning_rates):
                    # Create and compile model
                    model = create_uncompiled_model()
                    if i == 2: model.summary() # Keep summary for the 3rd model as in original
                    
                    hparams['LEARNING_RATE'] = lr
                    compile_model(model)
                    models.append(model)

                    # Train model
                    print(f"Training model n.{i+1} (LR: {lr})")
                    history = model.fit(
                        x=train_dataset_final,
                        epochs=hparams['EPOCHS'],
                        validation_data=validation_dataset_final,
                        verbose='auto' # Optional: reduces clutter during multi-model training
                    )
                    histories.append(history)

                print()
                plot_training_histories(histories)
            hparams['LEARNING_RATE'] = recomended_lr
        else: pass

        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae',    # Monitor the validation loss
            #min_delta = 0,
            patience=hparams['EARLY_STOP_PATIENCE'],            # Stop if val_loss doesn't improve for X epochs
            verbose=1,
            #mode='auto',
            #baseline=None,
            restore_best_weights=True # Restore model weights from the epoch with the best val_loss
            #start_from_epoch=0
        )

        learning_rate_scheduler = tf.keras.callbacks.ReduceLROnPlateau(   
            monitor='val_mae',
            factor=hparams['REDUCE_LR_FACTOR'],
            patience=hparams['REDUCE_LR_PATIENCE'],
            verbose=1,
            #mode='auto',
            #min_delta=0.0001,
            #cooldown=0,
            min_lr=hparams['REDUCE_LR_MIN_LR'],
            #**kwargs
        )

        if LOSS_FNC_TEST:
            # Define configurations
            loss_configs = [
                ("Huber_004", tf.keras.losses.Huber(delta=0.04)),
                ("Huber_005", tf.keras.losses.Huber(delta=0.05)),
                ("Huber_006", tf.keras.losses.Huber(delta=0.06)),
                ("MSE", tf.keras.losses.MeanSquaredError()),
                ("MAE", tf.keras.losses.MeanAbsoluteError()),
                #("LogCosh", tf.keras.losses.LogCosh()),
                #("Quantile", MyLossLib.QuantileLoss(quantile=0.9)),
                #("Dilate", MyLossLib.DilateLoss(alpha=0.8, gamma=0.5)),
                #("PatchStructural", MyLossLib.PatchStructuralLoss(patch_size=10)),
                #("ExtremePeak", MyLossLib.ExtremePeakLoss(alpha=3.0))
            ]

            all_histories = []

            models = []

            # Train each model
            for name, loss_fnc in loss_configs:
                # prevent memory leakage or weight carry-over between tests
                tf.keras.backend.clear_session()

                print(f"\n--- Testing Loss Function: {name} ---")
                hparams['LOSS_FUNCTION'] = loss_fnc
                
                model = create_uncompiled_model()
                model._name = name # Set the name for plotting
                compile_model(model)
                
                history = model.fit(
                    x=train_dataset_final,
                    #y=None,
                    #batch_size=None,
                    epochs=hparams['EPOCHS'],
                    verbose='auto',
                    callbacks=[ learning_rate_scheduler, early_stopping_callback], 
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
                
                # Tag the history object for our plotter
                history.user_defined_name = name
                all_histories.append(history)

                models.append(model)

            # Generate Comparison Table (The neutral third-party metric report)
            print("\n" + "="*50)
            print(f"{'Model Loss Function':<20} | {'Val MAE':<10} | {'Val RMSE':<10}")
            print("-" * 50)
            
            for h in all_histories:
                v_mae = min(h.history['val_mae'])
                v_rmse = min(h.history['val_rmse'])
                print(f"{h.user_defined_name:<20} | {v_mae:<10.4f} | {v_rmse:<10.4f}")
            print("="*50)

            # 4. Save the interactive chart
            plot_training_histories(all_histories, filename="Loss_Function_Comparison.html")

        else:
            pass 

        hparams['LOSS_FUNCTION'] = tf.keras.losses.Huber() 

        # Create final model
        print("Final model:")
        nn_model = create_uncompiled_model()
        nn_model.summary()
        compile_model(nn_model)

        window_offset = hparams['WINDOW_SIZE']

        actual_values = features_valid[window_offset:]
        model_names = [config[0] for config in loss_configs]

        final_predictions = []
        all_metric_results = []
        for i, model in enumerate(models):
            final_prediction = generate_model_predictions(
                model=model,               # The current model from your loop
                series=features_valid,     # The validation data series
                window_size=hparams['WINDOW_SIZE'],
                batch_size=hparams['BATCH_SIZE'],
                train_min=train_min, 
                train_max=train_max
            )
            
            # Get the number of predictions we actually have
            num_preds = len(final_prediction)
            
            # Slice the actual values and time from the END to match the number of predictions
            # This handles any off-by-one errors from windowing automatically
            current_actual_values = features_valid[-num_preds:]
            current_time_valid = time_valid[-num_preds:]
            
            # Unscale the actual values for metric calculation (using the same length)
            actual_unscaled = (current_actual_values * (train_max - train_min)) + train_min

            # Evaluate with perfectly matched lengths
            metrics = evaluate_predictions(actual_unscaled, final_prediction, model_names[i])
            all_metric_results.append(metrics)
            
            # Store for the dashboard
            final_predictions.append(final_prediction)

        generate_comparison_dashboard(
            time=current_time_valid,
            actual_series=actual_unscaled,
            predictions_list=final_predictions,
            metrics_list=all_metric_results,
            output_filename="Sunspot_Comparison_Dashboard.html"
        )

        exit()

        history_log_path = os.path.join(current_dir, "training_history.json")
        loss_acc_chart_path = os.path.join(current_dir, "training_loss_acc_chart.html")
        plot_training_charts(training_history, history_log_path, loss_acc_chart_path)

        SAVED_MODEL_PATH = os.path.join(current_dir, "trained_model_complete.tf")
        nn_model.save(SAVED_MODEL_PATH, save_format='tf')
        print("\nModel saved to {SAVED_MODEL_PATH}")


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