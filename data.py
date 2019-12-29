"""
Functions that get library data, clean data and explore the data.
"""

import os 
import pandas as pd
import numpy as np
import sqlite3

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sbn

def csv_to_sql_database(loc: str, db_name: str, con) -> None:
    """
    Converts csv files to sql database

    Args: 
        loc: location of csv files
        db_name: sql database name
        con: connection object to sqlite3 database

    Returns: None
    """    

    # Directory with data
    os.chdir(loc)

    # Save library data to SQL database
    for filename in os.listdir():
        if filename.endswith('csv'):
            # Load library data as dataframe
            lib_data = pd.read_csv(filename, encoding = 'latin-1', low_memory=False)
            # Save to SQL database
            filename_for_db = 'library_' + str(filename)[:4] 
            lib_data.to_sql(filename_for_db, con, if_exists='replace')

    return 

def get_data(loc: str, table_name_list: list, con) -> dict:
    """
    Queries list of annual library data tables. 
    Gets relevants data and transforms when necessary. 
    Calculates usage for each library.

    Args: 
        loc: location of sql database
        table_name_list: list of tables to query
        con: connection object to sqlite3 database

    Returns: 
        library_df_dict: dictionary of dataframes
    """   

    # Directory with SQL database
    os.chdir(loc)

    library_df_dict  = {}

    for table_name in table_name_list:
        _LIB_DATA_QUERY = """
        SELECT
            -- Library identifier 
            FSCSKEY,
            -- Normalize all features except hours_open by population 
            -- Rename features for clarity
            -- Multiply by 1.0 to ensure float
            benefit * 1.0/popu_lsa AS benefits, 
            bkmob * 1.0/popu_lsa AS book_mobile, 
            bkvol * 1.0/popu_lsa AS book_volume, 
            branlib * 1.0/popu_lsa AS branch_libraries, 
            capital * 1.0/popu_lsa AS capital_expenses,  
            centlib * 1.0/popu_lsa AS central_libraries,  
            ebook * 1.0/popu_lsa AS eBooks, 
            elmatexp * 1.0/popu_lsa AS electronic_material_expenses,  
            hrs_open AS hours_open,  
            kidpro * 1.0/popu_lsa AS kids_programs, 
            libraria * 1.0/popu_lsa AS librarians,  
            loanfm * 1.0/popu_lsa AS loans_received,  
            loanto * 1.0/popu_lsa AS loans_sent,  
            master * 1.0/popu_lsa AS librarians_w_masters,  
            othmatex * 1.0/popu_lsa AS other_material_expenses, 
            othopexp * 1.0/popu_lsa AS other_operational_expenses,  
            othpaid * 1.0/popu_lsa AS other_paid_staff,   
            prmatexp * 1.0/popu_lsa AS print_material_expenses,  
            -- calculate other_programs from total programs
            (totpro - (yapro + kidpro)) * 0.1/popu_lsa AS other_programs,
            salaries * 1.0/popu_lsa AS salaries,  
            yapro * 1.0/popu_lsa AS young_adult_programs,  
            
            -- Combine downloadable and physical units to single measure
            audio_dl + audio_ph AS audio,
            video_dl + video_ph AS video,
            
            -- Calculate usage (label to be predicted)
            ROUND(CAST(visits AS FLOAT)/popu_lsa, 2) AS usage
        FROM 
            {}
        WHERE 
            -- Remove any rows where Usage can't be calculated or is 0 (rare)
            usage IS NOT NULL
            AND usage != 0
            -- Remove any rows where POPU_LSA value is missing (sometimes denoted with negative values)
            AND POPU_LSA > 0
        
        """.format(table_name)
        
        # Save library df as value in dict
        library_df_dict[table_name] = pd.read_sql_query(_LIB_DATA_QUERY, con)

    return library_df_dict

def concatenate_data(dataframe_dict: dict) -> pd.DataFrame:
    """
    Concatenates all dataframes stored in dict. 
    
    Args:
        dataframe_dict: dictionary of dataframes to concatenate
    
    Returns: 
        data_concat: All data concatenated to single dataframe
    
    """
    
    data_concat = pd.concat(list(dataframe_dict.values()), axis=0, sort=False)
    
    return data_concat

def impute_nulls(df: pd.DataFrame, max_null: int) -> pd.DataFrame:
    """
    Remove rows with more than given number of nulls.
    Impute remaining nulls with feature means. 
    
    Args:
        df: features dataframe
        max_null: maximum number of nulls allowed per sample (row)

    Returns: 
        df_imputed: dataframe     
    """ 

    # Remove FSCKKEY before filling unknown values
    FSCSKEY_vals = df.FSCSKEY 
    df = df.drop(columns='FSCSKEY')
    
    # Convert negative values to null 
    # -1, -3 and -9 used in original data to indicate unknown. 
    # Any other negative values are nonsensical
    df[df < 0] = np.nan

    # Add back FSCSKEY
    df['FSCSKEY'] = FSCSKEY_vals
    
    # Sum nulls across each row
    df['sum_nulls'] = df.isnull().sum(axis=1)
    
    # Remove rows with more nulls than max_null
    df_reduced = df[df.sum_nulls <= max_null]
    df_reduced = df_reduced.drop(columns='sum_nulls')
    
    # Impute remaining nulls with feature mean
    df_imputed = df_reduced.fillna(df_reduced.mean())
    
    return df_imputed  

def remove_usage_outliers(df: pd.DataFrame, std_multiplier: int) -> pd.DataFrame:
    """
    Remove usage outliers from dataframe
    
    Args:
        df: features dataframe
        std_multiplier: how many standard deviations away from mean is considered outlier
    
    Returns:
        df_no_outliers: dataframe without usage outliers
    """
    # Calculate outlier bounds
    high_outlier = df.usage.mean() + std_multiplier*(df.usage.std())
    low_outlier = df.usage.mean() - std_multiplier*(df.usage.std())
    
    # Remove samples where usage is outlier
    df_no_outliers = df[(df.usage < high_outlier) & (df.usage > low_outlier)]
    
    return df_no_outliers

def clean_raw_data(dataframe_dict: dict, max_null: int, std_multiplier: int) -> pd.DataFrame:
    """
    Concatenate all dataframes stored in dict.
    Remove rows with more than given number of nulls.
    Impute remaining nulls with feature means. 
    Remove usage outliers from dataframe.
    
    Args:
        dataframe_dict: dictionary of dataframes to concatenate
        max_null: maximum number of nulls allowed per sample (row)
        std_multiplier: how many standard deviations away from mean is considered outlier
    
    Returns: 
        clean: concatenated data with nulls imputed and usage outliers removed
    """
    concat_df = concatenate_data(dataframe_dict)
    impute_df = impute_nulls(concat_df, max_null)
    clean_df = remove_usage_outliers(impute_df, std_multiplier)

    return clean_df


def usage_plot(usage_list: list, num_bins: int = 500) -> None:
    """
    Plot histogram of library usage

    Args:
        usage_list: list of usage values
        num_bins: number of bins in which to bin usage values.

    Returns: None

    """
    fig, ax = plt.subplots(1, 1, figsize = (9,5))
    plt.rcParams.update({'font.size': 20})
    
    #plt.axvline(0, color='black', lw = 2)
    #plt.axhline(0, color='black', lw = 2)
    plt.ylabel("y label")
    ax.hist(usage_list, bins = num_bins, color = "black")
    #ReducedDF['Usage'].plot.hist(grid = False, color = "black", bins = 5000, xlim = (0, 55))
    ax.set_xlabel("Usage")
    ax.set_ylabel("Library frequency")
    plt.show()

    return

def usage_corr_plot(df: pd.DataFrame) -> None:
    """
    Plot correlation between usage and features

    Args:
        df: features dataframe (with usage label)

    Returns: None

    """

    # Get correlations between usage and all other features
    corr_w_usage = df.corr().usage.sort_values()
    
    # Plot all correlation except usage with itself
    fig, ax = plt.subplots(1, 1, figsize = (9,5))
    plt.rcParams.update({'font.size': 14})
    ax.bar(corr_w_usage.index[:-1], corr_w_usage[:-1], color='black')
    ax.xaxis.set_tick_params(rotation=90)
    ax.set_xlabel("Features")
    ax.set_ylabel("Correlation with usage")
    plt.show()

    return

def feat_corr_plot(df: pd.DataFrame, chosen_figsize: tuple = (10, 8)) -> None:
    """
    Plots correlation matrix
    
    Args:
        df: dataframe of features
        chosen_figsize: width and height of figure as tuple

    Returns: None
    """

    corr = df.corr()

    fig, ax = plt.subplots(1, 1, figsize=chosen_figsize)
    plt.rcParams.update({'font.size': 12})

    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    sbn.heatmap(corr, mask=mask)
    plt.show()
    
    return

