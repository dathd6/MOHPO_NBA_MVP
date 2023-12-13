# DEFAULT VARIABLES
YEARS = range(2003, 2023)

# FEATURES
YEAR_FEATURE = 'Year'
PLAYER_NAME_FEATURE = 'Player'
MVP_SHARE_FEATURE = 'Share'
MVP_RANKING_FEATURE = 'MVP Ranking'
RMSE_METRIC = 'RMSE'
ACCURACY_METRIC = 'ACCURACY'

# DATASET VARIABLES
REPORT_BEST_PARAMS = 'best_params.csv'
DATASET_FOLDER = 'data'
STANDARD_DATASET_NAME = 'player_data_03_22.csv'
ADVANCED_DATASET_NAMES = [f'advanced_stats_{year}.csv' for year in YEARS]

# DATA CLEANING
FILTER_ADVANCED_FEATURES = ['Player', 'Year', 'PER', 'TS%', 'WS', 'BPM', 'VORP', 'USG%']
DROP_STANDARD_FEATURES = ['0', 'FG', '3P', '2PA', '2P', '2P%', 'FT', 'L', 'PS/G', 'PA/G', 'Pts Won', 'Pts Max', 'G', 'Tm', 'SRS', 'Pos', 'TOV', 'PF', 'GB', 'Team', 'Age', 'MP', 'ORB', 'DRB']
WEIRD_CHARACTER_MAPPING = [
    ('Ä‡', 'ć'),
    ('Ä', 'č'),
    ('Ã³', 'ó'),
    ('Ã¶', 'ö'),
    ('Ã¼', 'ü'),
    ('ÄŸ', 'ğ')
]
MIMIMUM_CRITERIA_FEATURES = ['PTS', 'GS', 'FG%', 'FGA', 'TRB', 'AST']

# MOEA
CROSSOVER_RATE = .9
MUTATION_RATE = .9
N_ESTIMATORS = [1, 10, 20, 50, 100]
LEARNING_RATE = [0, 0.1, 0.2, 0.5, 1]
