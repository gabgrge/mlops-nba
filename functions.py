import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
from typing import Tuple, Optional
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


RAW_DATA_DIR = Path('data/raw')
CURATED_DATA_DIR = Path('data/curated')
MODELS_DIR = Path('data/models')


def create_nba_features(players: pd.DataFrame) -> pd.DataFrame:
    """Create features from raw data."""
    players["EFF"] = (
        players.PTS
        + players.TRB
        + players.AST
        + players.STL
        + players.BLK
        - (players.FGA - players.FG)
        - (players.FTA - players.FT)
        - players.TOV
    )

    players['TS%'] = np.where(
        (2 * (players['FGA'] + 0.44 * players['FTA'])) != 0,
        players['PTS'] / (2 * (players['FGA'] + 0.44 * players['FTA'])),
        0
    )

    players["position"] = players.Pos.map(
        {"PG": "Backcourt", "SG": "Backcourt", "SF": "Wing", "SF-PF": "Wing", "PF": "Big", "C": "Big", }
    )

    return players


def stars_definition(row: pd.Series, EFFICENCY_THRESHOLD: int = 12, POINTS_THRESHOLD: int = 10, AGE_THRESHOLD: int = 23) -> bool:
    """Define rising stars."""
    return (
        row["EFF"] >= EFFICENCY_THRESHOLD
        and row["PTS"] >= POINTS_THRESHOLD
        and row["Age"] <= AGE_THRESHOLD
    )


def collect_raw_data() -> bool:
    """Collect data from raw data directory and save it in curated data directory."""
    print("Collecting raw data...")

    files = list(RAW_DATA_DIR.glob('*.csv'))
    pattern = re.compile(r'\d{4}-\d{4} NBA Player Stats - Regular\.csv$')
    matching_files = sorted(f for f in files if pattern.match(f.name))

    if len(matching_files) == 0:
        print("No data found.")
        return False

    filename = matching_files[-1]
    print(f"Running on file: {filename}")

    players = pd.read_csv(filename, encoding='Windows-1252')
    players = create_nba_features(players=players)
    players["rising_stars"] = players.apply(stars_definition, axis=1)

    name = filename.stem
    season = re.search(r"\d{4}-\d{4}", name).group(0)

    if not (CURATED_DATA_DIR / (season + "_players.csv")).exists():
        players.to_csv(str(CURATED_DATA_DIR / (season + "_players.csv")), encoding='Windows-1252')
        print("Saved data to: " + str(CURATED_DATA_DIR / (season + "_players.csv")))
        return True
    else:
        print("No new data found.")
        return False


def collect_curated_data() -> Tuple[pd.DataFrame, str]:
    """Collect data from curated data directory."""
    print("Collecting curated data...")

    files = list(CURATED_DATA_DIR.glob('*.csv'))
    pattern = re.compile(r'\d{4}-\d{4}_players\.csv$')
    matching_files = sorted(f for f in files if pattern.match(f.name))

    if len(matching_files) == 0:
        print("No data found.")
        return pd.DataFrame(), ""

    filename = matching_files[-1]
    season = re.search(r"\d{4}-\d{4}", filename.stem).group(0)
    print(f"Loading file: {filename}")

    players = pd.read_csv(filename, encoding='Windows-1252')

    return players, season


def create_model(players: pd.DataFrame, season: str) -> None:
    """Create a model to predict whether a player is a future star or not."""
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(str(MODELS_DIR / (season + "_model.log")))
    handler.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -\n%(message)s')
    handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    print("Creating model...")

    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(),
             ['Age', 'G', 'GS', 'MP', 'FG', 'FGA', '3P', '3PA', '2P', '2PA', 'FT', 'FTA', 'ORB', 'DRB', 'TRB',
              'AST', 'STL', 'BLK', 'TOV', 'PF']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Pos', 'Tm'])
        ])

    # Define model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])

    # Split data into training and test sets
    X = players.drop(['Player', 'rising_stars'], axis=1)
    y = players['rising_stars']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train the model
    pipeline = model.fit(X_train, y_train)
    logger.info(f"Model pipeline:\n{pipeline}")

    # Save the model to a file
    joblib.dump(pipeline, str(MODELS_DIR / (season + "_model.joblib")))

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy}")

    # Confusion matrix
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"Mean cross-validation score: {np.mean(cv_scores)}")

    # Learning curves
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)

    plt.figure(figsize=(12, 8))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training score')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-validation score')
    plt.title('Learning curve')
    plt.xlabel('Training set size'), plt.ylabel('Accuracy score'), plt.legend(loc='best')
    plt.savefig(str(MODELS_DIR / (season + "_learning_curve.png")))


def load_model() -> Optional[Pipeline]:
    """Load a model from a file."""
    print("Loading model...")

    files = list(MODELS_DIR.glob('*.joblib'))
    pattern = re.compile(r'\d{4}-\d{4}_model\.joblib$')
    matching_files = sorted(f for f in files if pattern.match(f.name))

    if len(matching_files) == 0:
        print("No model found.")
        return None

    filename = matching_files[-1]
    model = joblib.load(filename)
    print(f"Loaded model: {filename.stem}")

    return model
