import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import typer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os


RAW_DATA_FOLDER = Path('/work/NLP/data/raw')
PROCESSED_DATA_FOLDER = Path('/work/NLP/data/processed')
FILE_PATHS = [
    RAW_DATA_FOLDER / 'RandomSample_1950.csv',
    RAW_DATA_FOLDER / 'RandomSample_1960.csv',
    RAW_DATA_FOLDER / 'RandomSample_1970.csv',
    RAW_DATA_FOLDER / 'RandomSample_1980.csv',
    RAW_DATA_FOLDER / 'RandomSample_1990.csv',
    RAW_DATA_FOLDER / 'RandomSample_2000.csv',
    RAW_DATA_FOLDER / 'RandomSample_2010.csv',
    RAW_DATA_FOLDER / 'RandomSample_2020.csv'
]


# --------------------------------------------------
# Preprocessing Pipeline
# --------------------------------------------------
class LyricsDataset:
    """Custom dataset for preprocessing song lyrics."""
    def __init__(self, data_paths: list[Path]) -> None:
        self.data_paths = data_paths
        
        try:
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        except LookupError:
            import nltk
            nltk.download('stopwords')
            nltk.download('wordnet')
            self.stop_words = set(stopwords.words('english'))
        
        self.strings_to_drop = [
            '*deleted lyrics*', '(Lyrics to be added once the project comes out)', '[Instrumental Track]',
            'Lyrics for this song have yet to be released. Please check back once the song has been released',
            '[Lyrics available upon release of the song]', '(Lyrics Pending)', '(Not Released)', 
            '[Lyrics coming soon]', 'Not Transcribed', '[Unreleased]', 
            'The lyrics for this song have yet to be transcribed',
            '[coming soon]', 'Unfinished....', '[Chorus]', '(Missing Lyrics)',
            'Unreleased', '(Lyrics Pending)', '(Not Released)',
            '© Universal Music Publishing Group'
        ]
        self.label_encoder = LabelEncoder()

    def merge_datasets(self) -> pd.DataFrame:
        """Loads and merges datasets."""
        dataframes = [pd.read_csv(path) for path in self.data_paths]
        combined_df = pd.concat(dataframes, ignore_index=True)
        return combined_df

    def data_cleaning(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """remvove duplicated lyrics, and keep only lyrics and year"""
        cleaned_df = combined_df[~combined_df['lyrics'].isin(self.strings_to_drop)]
        
        cleaned_df = cleaned_df.sort_values(by='year').drop_duplicates(subset='lyrics', keep='first')
        
        columns_to_keep = ['lyrics', 'year']
        cleaned_df = cleaned_df[columns_to_keep]

        return cleaned_df

    def preprocess_lyrics(self, lyrics: str) -> str:
        """Cleans and lemmatizes lyrics."""
        common_words_to_exclude = [
            r'yeah', r'o+h+', r'u+h+', r'uhh', r'o{3,}', r'o{2,}h+',
            r'a+h+', r'l+a+', r'n+a+', r'e+h+', r'h{2,}m+', r'm{3,}', 
            r'ooh', r'hah', r'u+g+h+'
        ]
  
        lyrics = re.sub(r'\[.*?\]', '', lyrics)
        lyrics = re.sub(r'\(.*?\)', '', lyrics)
        lyrics = re.sub(r'[^\w\s]', '', lyrics)
  
        words = lyrics.split()
        processed_lyrics = []
  
        for word in words:
            if word.lower() not in self.stop_words and word.isalpha():
                exclude_word = any(re.match(pattern, word.lower()) for pattern in common_words_to_exclude)
                if not exclude_word:
                    processed_lyrics.append(self.lemmatizer.lemmatize(word.lower()))
  
        return ' '.join(processed_lyrics)

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms years to decades, encodes them, and adds processed lyrics"""
        
        df['decade'] = df['year'].apply(lambda x: int(x / 10) * 10)
        df = df.drop('year', axis=1, errors='ignore')
        df['decade'] = self.label_encoder.fit_transform(df['decade'])

        df['lyrics_processed'] = df['lyrics'].apply(self.preprocess_lyrics)
        return df
    
    def split_dataset(self, df: pd.DataFrame):
        """Split into train, test, val sets (70%, 20%, 10%)."""

        target_column = 'decade'
        X = df.drop(target_column, axis=1, errors='ignore')
        y = df[target_column]
    
        X_train, X_remaining, y_train, y_remaining = train_test_split(
            X, y, test_size=0.3, random_state=42
        ) 
        X_test, X_val, y_test, y_val = train_test_split(
            X_remaining, y_remaining, test_size=0.333, random_state=42
        )

        return X_train, y_train, X_test, y_test, X_val, y_val
    
    def save_splits(self, 
                    X_train: pd.DataFrame, y_train: pd.DataFrame,
                    X_test: pd.DataFrame, y_test: pd.DataFrame,
                    X_val: pd.DataFrame, y_val: pd.DataFrame,
                    output_folder: Path, dataset_type: str) -> None:

        X_train.to_csv(output_folder / f'X_train_{dataset_type}.csv', index = False )
        y_train.to_csv(output_folder / f'y_train_{dataset_type}.csv', index = False )
        X_test.to_csv(output_folder / f'X_test_{dataset_type}.csv', index = False )
        y_test.to_csv(output_folder / f'y_test_{dataset_type}.csv', index = False )
        X_val.to_csv(output_folder / f'X_val_{dataset_type}.csv', index = False )
        y_val.to_csv(output_folder / f'y_val_{dataset_type}.csv', index = False )
        
    def save_raw_dataset(self, output_folder: Path) -> None:
        """Merges and saves the raw dataset without preprocessing."""

        combined_df = self.merge_datasets()
        output_path = output_folder / 'raw_dataset.csv'
        combined_df.to_csv(output_path, index = False  )
        print(f"Raw dataset saved to {output_path}.")
    
    def preprocess(self, output_folder: Path) -> None:
        """
        Merges, cleans, applies feature engineering, then
        saves two DataFrames: one with processed lyrics, one with the original lyrics.
        """

        combined_df = self.merge_datasets()
        cleaned_df = self.data_cleaning(combined_df)
        final_df = self.feature_engineering(cleaned_df)
    
        processedlyrics_df = final_df.drop('lyrics', axis=1, errors='ignore')
        unprocessedlyrics_df = final_df.drop('lyrics_processed', axis=1, errors='ignore')

        processed_path = output_folder / 'processed_dataset.csv'
        unprocessed_path = output_folder / 'unprocessed_dataset.csv'

        processedlyrics_df.to_csv(processed_path, index = False )
        unprocessedlyrics_df.to_csv(unprocessed_path, index = False )

        print(f"Processed dataset saved to {processed_path}")
        print(f"Unprocessed dataset saved to {unprocessed_path}")

    def preprocess_split(self, output_folder: Path) -> None:
        """Runs the full pipeline (merge, clean, feature-engineer) and splits into train/test/val."""

        combined_df = self.merge_datasets()
        cleaned_df = self.data_cleaning(combined_df)
        final_df = self.feature_engineering(cleaned_df)
        X_train, y_train, X_test, y_test, X_val, y_val = self.split_dataset(final_df)

        X_train_processed = X_train.drop('lyrics', axis=1, errors='ignore')
        X_test_processed  = X_test.drop('lyrics', axis=1, errors='ignore')
        X_val_processed   = X_val.drop('lyrics', axis=1, errors='ignore')

        X_train_unprocessed = X_train.drop('lyrics_processed', axis=1, errors='ignore')
        X_test_unprocessed  = X_test.drop('lyrics_processed', axis=1, errors='ignore')
        X_val_unprocessed   = X_val.drop('lyrics_processed', axis=1, errors='ignore')

        self.save_splits(
            X_train_processed, y_train,
            X_test_processed,  y_test,
            X_val_processed,   y_val,
            output_folder,     "processed"
        )

        self.save_splits(
            X_train_unprocessed, y_train,
            X_test_unprocessed,  y_test,
            X_val_unprocessed,   y_val,
            output_folder,       "unprocessed"
        )


# --------------------------------------------------
# Create different datasets according to needs 
# --------------------------------------------------

def create_raw_dataset() -> None:
    # for exploratory data analysis
    dataset = LyricsDataset(FILE_PATHS)
    dataset.save_raw_dataset(RAW_DATA_FOLDER)

def preprocessed_dataset() -> None:
    # for BERT 
    dataset = LyricsDataset(FILE_PATHS)
    dataset.preprocess(PROCESSED_DATA_FOLDER)

def split_preprocessed_dataset() -> None:
    # for the other models 
    dataset = LyricsDataset(FILE_PATHS)
    dataset.preprocess_split(PROCESSED_DATA_FOLDER)

def main():
    split_preprocessed_dataset()

if __name__ == "__main__":
    typer.run(main)

