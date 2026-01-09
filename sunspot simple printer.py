import pandas as pd
import plotly.express as px
import os

def generate_sunspot_chart():
    # 1. Define the file path
    # We use 'r' before the string to handle backslashes correctly in Windows
    file_path = r'C:\Repositories\Daily-Sun-Spot-Data\dataset\Sunspots_linear_extrapolation.csv'
    
    # Check if the file exists before attempting to read
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # 2. Read the CSV file
        df = pd.read_csv(file_path)
        
        # Specified column name
        column_name = 'Number of Sunspots filled'
        
        if column_name not in df.columns:
            print(f"Error: The column '{column_name}' does not exist in the CSV.")
            print("Available columns:", df.columns.tolist())
            return

        # 3. Create the interactive chart
        # Use 'Date' for the X-axis if it exists; otherwise, use the dataframe index
        x_axis = 'Date' if 'Date' in df.columns else df.index
        
        fig = px.line(
            df, 
            x=x_axis, 
            y=column_name,
            title='Daily Sunspot Evolution (Linear Extrapolation)',
            labels={column_name: 'Sunspot Number', 'Date': 'Date', 'index': 'Day Index'},
            template='plotly_white'
        )

        # Enhance interactivity and layout
        fig.update_layout(
            hovermode="x unified",
            xaxis_title="Timeline",
            yaxis_title="Sunspot Count",
            showlegend=False
        )

        # 4. Save as a standalone HTML file
        output_file = "Sunspot_Interactive_Chart.html"
        fig.write_html(output_file)
        
        print(f"Success! The interactive chart has been saved as: {output_file}")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    generate_sunspot_chart()