import pandas as pd
from utils import state_dict, industry_dict, convert_sic_to_naics
import bls
import requests
import os
import re

def get_yearly_averaged_value(
        series: pd.DataFrame
        ) -> pd.DataFrame:
    '''
    Calculate the yearly averaged values from a given time series dataframe.
    
    Parameters:
    series (pd.DataFrame): A dataframe containing the time series data.
    
    Returns:
    pd.DataFrame: A dataframe with the yearly averaged values.
    '''
    # Resample the data to yearly frequency and calculate the mean for each year
    df_yearly = series.resample('Y').mean()
    
    # Rename the index to 'Year' for clarity
    df_yearly.index.name = 'Year'
    
    return df_yearly

def get_series_data_by_state(
        start_year: int,
        end_year: int,
        api_key: str) -> list[pd.DataFrame]:
    """
    Fetches and processes unemployment rate data for each state from the BLS API.

    Args:
        start_year (int): The starting year for the data.
        end_year (int): The ending year for the data.

    Returns:
        list: A list of DataFrames, each containing the yearly averaged unemployment rate data for a state.
    """
    state_unemployment_rates = []
    for i in range(1, 57):
        if (i != 3) & (i != 7) & (i != 14) & (i != 43) & (i != 52):
            print(i)
            state_id = str(i).zfill(2)
            series_name = f'LASST{state_id}0000000000003'
            df = bls.get_series(series_name, start_year, end_year, api_key)
            yearly_data = get_yearly_averaged_value(df).to_frame().reset_index()
            state = state_dict[i]
            yearly_data['State'] = state
            yearly_data.rename(columns={series_name: 'Unemployment Rate'}, inplace=True)
            state_unemployment_rates.append(yearly_data)
    return state_unemployment_rates
    
def get_establishment_count_data(year):
    """
    Fetches establishment count data from the Census Bureau API for a specific year.

    Args:
        year (int): The year for which to fetch the data.

    Returns:
        pd.DataFrame: A DataFrame containing the establishment count data with columns:
                      'Year', 'Industry', 'State', and 'Establishments'.
    """
    if year >= 2017:
        url = f"https://api.census.gov/data/{year}/cbp?get=ESTAB,STATE&for=state:*&NAICS2017=*"
    elif year >= 2012:
        url = f"https://api.census.gov/data/{year}/cbp?get=ESTAB,NAICS2012_TTL&for=state:*&NAICS2012=*"
    elif year > 2007:
        url = f"https://api.census.gov/data/{year}/cbp?get=ESTAB,NAICS2007_TTL&for=state:*&NAICS2007=*"
    elif year > 2002:
        url = f"https://api.census.gov/data/{year}/cbp?get=ESTAB,NAICS2002_TTL&for=state:*&NAICS2002=*"
    elif year > 1997:
        url = f"https://api.census.gov/data/{year}/cbp?get=ESTAB,NAICS1997_TTL&for=state:*&NAICS1997=*"
    else:
        url = f"https://api.census.gov/data/{year}/cbp?get=ESTAB,SIC_TTL&for=state:*&SIC=*"
    
    response = requests.get(url)
    data = response.json()
    columns = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=columns)
    df = df.loc[:, ~df.columns.duplicated()]
    
    # Convert NAICS or SIC codes to NAICS 2017 standard
    if year >= 2017:
        df['NAICS2017'] = df['NAICS2017']
    elif year >= 2012:
        df.rename(columns={'NAICS2012': 'NAICS2017'}, inplace=True)
    elif year > 2007:
        df.rename(columns={'NAICS2007': 'NAICS2017'}, inplace=True)
    elif year > 2002:
        df.rename(columns={'NAICS2002': 'NAICS2017'}, inplace=True)
    elif year > 1997:
        df.rename(columns={'NAICS1997': 'NAICS2017'}, inplace=True)
    else:
        df['SIC'] = df['SIC'].str.strip()
        df['SIC'] = df['SIC'].apply(lambda x: convert_sic_to_naics(x) if len(x) == 2 else None)
        df.dropna(subset=['SIC'], inplace=True)
        df.rename(columns={'SIC': 'NAICS2017'}, inplace=True)
    
    
    # Filter for valid NAICS codes and state codes
    df['NAICS2017'] = df['NAICS2017'].str.strip()
    df_filtered = df[df["NAICS2017"].str.match(r'^\d{2}\s*$|^\d{2}-\d{2}\s*$')]
    df_filtered = df_filtered[df_filtered['state'].astype('int') <= 56]
    
    # Map state codes and industry codes
    df_filtered['State'] = df_filtered['state'].astype('int').map(state_dict)
    df_filtered['Industry'] = df_filtered['NAICS2017'].map(industry_dict)
    
    # Clean up the DataFrame
    df_filtered.drop(columns=['NAICS2017'], inplace=True)
    df_filtered.rename(columns={'ESTAB': 'Establishments'}, inplace=True)
    df_filtered["Year"] = year
    
    return df_filtered[['Year', 'Industry', 'State', 'Establishments']].reset_index(drop=True)

def download_bls_table(url, output_dir="bls_data", filename=None):
    """
    Downloads a BLS table from a given URL and saves it as a text file.

    Args:
        url: The URL of the BLS table (text file).
        output_dir: The directory to save the downloaded file.
        filename: (Optional) Custom filename. If None, uses the filename from the URL.

    Returns:
        The full path to the saved file, or None if an error occurred.
    """

    try:
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'}
        # Fetch the file content
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Determine the filename
        if filename is None:
            # Extract filename from URL
            filename = os.path.basename(url)
            # Further improve the filename: remove any query parameters, etc.
            filename = filename.split('?')[0] # remove query parameters, if any

        # Ensure a .txt extension (or appropriate for the content)
        if not filename.lower().endswith(('.txt', '.csv', '.dat')):  # Add other likely extensions
             filename += '.txt' # Append .txt if no suitable extension is found


        # Construct the full file path
        file_path = os.path.join(output_dir, filename)

        # Save the file content
        with open(file_path, 'w', encoding='utf-8') as f:  # Use UTF-8 encoding
            f.write(response.text)

        print(f"Downloaded and saved to: {file_path}")
        return file_path

    except requests.exceptions.RequestException as e:
        print(f"Error downloading from {url}: {e}")
        return None
    except OSError as e:
        print(f"Error saving file: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

def parse_survival_text_file(
        file_path: str,
        ) -> pd.DataFrame:
    """
    Parses BLS text file on business survival data and extracts relevant data into a DataFrame.

    Args:
        file_path (str): The path to the text file to be parsed.

    Returns:
        pd.DataFrame: A DataFrame containing the parsed data with columns:
            'Year Established', 'Year', 'Surviving Establishments', 
            'Total Employment of Survivors', 'Survival Rates Since Birth', 
            'Survival Rates of Previous Year's Survivors', 'Average Employment of Survivors'.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = []
    current_year = None

    for line in lines:
        # Check for the "Annual openings Year ended:" line to get the current year
        year_match = re.match(r'Year ended: March (\d{4})', line)
        if year_match:
            current_year = year_match.group(1)
            continue

        # Skip lines that do not contain data
        if not re.match(r'\s*March \d{4}', line):
            continue

        # Extract data from the line
        columns = re.split(r'\s{2,}', line.strip())
        if len(columns) == 6:
            columns.insert(0, current_year)  # Insert the current year at the beginning
            columns[1] = columns[1][-4:]
            data.append(columns)

    # Create a DataFrame from the parsed data
    df = pd.DataFrame(data, columns=[
        'Year Established', 'Year', 'Surviving Establishments', 'Total Employment of Survivors',
        'Survival Rates Since Birth', 'Survival Rates of Previous Year\'s Survivors',
        'Average Employment of Survivors'
    ])

    return df



