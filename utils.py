# Dependencies and packages
import pandas as pd
import numpy as np
# Constant variables
from constants import YEARS
from constants import DATASET_FOLDER
from constants import WEIRD_CHARACTER_MAPPING
from constants import MIMIMUM_CRITERIA_FEATURES
from constants import STANDARD_DATASET_NAME, ADVANCED_DATASET_NAMES
from constants import FILTER_ADVANCED_FEATURES, DROP_STANDARD_FEATURES
from constants import YEAR_FEATURE, \
                      PLAYER_NAME_FEATURE, \
                      MVP_SHARE_FEATURE, \
                      MVP_RANKING_FEATURE

PER_FEATURE = 'PER'

def extract_nba_player_dataset():
    # Import Advanced Statistics
    advanced_stats = pd.DataFrame()
    # Get advanced stats for each year
    for i, dataset_name in enumerate(ADVANCED_DATASET_NAMES):
        advanced_stats_by_year = pd.read_csv(f'{DATASET_FOLDER}/{dataset_name}')
        advanced_stats_by_year[YEAR_FEATURE] = np.array([
            YEARS[i] for _ in range(len(advanced_stats_by_year))
        ]) # Add year column
        advanced_stats = pd.concat([advanced_stats, advanced_stats_by_year], ignore_index=True)
    # Filter only necessary columns
    advanced_stats = advanced_stats[FILTER_ADVANCED_FEATURES]
    advanced_stats[PLAYER_NAME_FEATURE] = advanced_stats[PLAYER_NAME_FEATURE].str.replace('*', '') # Remove weird character in player name

    # Import Standard Statistics
    nba_stats = pd.read_csv(f'{DATASET_FOLDER}/{STANDARD_DATASET_NAME}')
    # Filter only necessary columns
    nba_stats = nba_stats.drop(DROP_STANDARD_FEATURES, axis=1)
    # Replace some character of the wrong Player name (mostly intertional player)
    for weird_character in WEIRD_CHARACTER_MAPPING:
        nba_stats[PLAYER_NAME_FEATURE] = nba_stats[PLAYER_NAME_FEATURE].str.replace(*weird_character)

    # Merge advanced stats with player standard stats
    nba_stats = nba_stats.merge(advanced_stats, how='left', on=[PLAYER_NAME_FEATURE, YEAR_FEATURE])
    nba_stats = nba_stats.sort_values(by=[YEAR_FEATURE, MVP_SHARE_FEATURE, PER_FEATURE], ascending=[True, False, False])

    # Get minimum criteria to filter the dataset
    MVP_player_each_season_index = nba_stats.groupby(YEAR_FEATURE)[MVP_SHARE_FEATURE].idxmax()
    minimum_criteria = nba_stats.loc[MVP_player_each_season_index].describe()[MIMIMUM_CRITERIA_FEATURES].loc['min',:]
    features = minimum_criteria.keys()
    filter_criteria = None
    for i in range(1, len(minimum_criteria)):
        if i == 1: 
            filter_criteria = (nba_stats[features[i]] >= minimum_criteria[i]) & \
                              (nba_stats[features[i - 1]] >= minimum_criteria[i - 1])
        filter_criteria = filter_criteria & (nba_stats[features[i]] >= minimum_criteria[i])
    potential_mvp_df = nba_stats[
        filter_criteria | (nba_stats[MVP_SHARE_FEATURE] > 0) # Give players that have been voted a chance
    ].reset_index(drop=True)

    # Ranking MVP by MVP Share
    potential_mvp_df[MVP_RANKING_FEATURE] = 0
    year = YEARS[0]
    rank = 1
    for index, row in potential_mvp_df.iterrows():
        if row[YEAR_FEATURE] > year:
            year = row[YEAR_FEATURE]
            rank = 1
        potential_mvp_df[MVP_RANKING_FEATURE][index] = rank
        rank += 1
    
    return potential_mvp_df

def generate_random_number(hp_range, is_int=False):
    block = 1 / len(hp_range)
    rate = np.random.rand()
    for index in range(1, len(hp_range)):
        if block * index <= rate and rate <= block * (index + 1):
            if is_int:
                return np.random.randint(hp_range[index - 1], hp_range[index])
            return round(
                np.random.uniform(hp_range[index - 1], hp_range[index]),
                4
            )
    return np.random.choice(hp_range)

def euclidean_distance_from_point_to_vector(point, start, end):
    point = np.array(point)
    start = np.array(start)
    end = np.array(end)

    # Calculate the vector representing the line
    vector = end - start

    distance = np.inf
    if len(point) == 2:
        # Calculate the parameters A, B, and C for the equation of the line
        A = -vector[1]
        B = vector[0]
        C = -(A * start[0] + B * start[1])

        # Calculate the distance using the formula
        distance = np.abs(A * point[0] + B * point[1] + C) / np.sqrt(A**2 + B**2)
    elif len(point) == 3:
        # Calculate the parameters A, B, and C for the equation of the plane
        A, B, C = np.cross(vector, point - start)
        D = -(A * start[0] + B * start[1] + C * start[2])

        # Calculate the distance using the formula
        distance = np.abs(A * point[0] + B * point[1] + C * point[2] + D) / np.sqrt(A**2 + B**2 + C**2)

    return distance
