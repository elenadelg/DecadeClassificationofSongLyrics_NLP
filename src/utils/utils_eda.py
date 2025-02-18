import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class ExploratoryDataAnalysis:
    """Class for performing exploratory data analysis on a dataset."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe

    def overview(self) -> None:
        """Prints an overview of the dataset, including its size, basic statistics, and information."""
        print("-" * 200)
        print("Overview of Dataset\n")
        print(f'The dataset contains {self.df.shape[0]} songs and {self.df.shape[1]} variables\n')
        print(self.df.describe())
        print("\n")
        self.df.info()

    def variable_summary(self) -> None:
        """Prints a summary of each variable, including datatype, unique values, missing values, and duplicates."""
        print("-" * 200)
        print("Overview of Variables\n")

        for column in self.df.columns:
            unique_values = self.df[column].nunique()
            missing_values = self.df[column].isnull().sum()
            duplicate_values = self.df[column].duplicated().sum()

            print(f"Variable: {column}")
            print(f"  - Data type: {self.df[column].dtype}")
            print(f"  - Unique values: {unique_values}")
            print(f"  - Missing values: {missing_values}")
            print(f"  - Duplicated values: {duplicate_values}")
            print("\n")

        # Special handling for 'language' and 'tag' columns
        if 'language' in self.df.columns:
            print("Value counts for 'language':")
            print(self.df['language'].value_counts())
            print("\n")

        if 'tag' in self.df.columns:
            print("Value counts for 'tag':")
            print(self.df['tag'].value_counts())
            print("\n")

    def class_balance(self) -> None:
        """Displays the distribution of songs over the years and per decade side by side."""
        # Compute decades
        self.df["decade"] = self.df["year"].apply(lambda x: int(x / 10) * 10)
        decade_counts = self.df["decade"].value_counts().sort_index()

        # Set up subplots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Song distribution over years
        sns.histplot(data=self.df, x="year", color='steelblue', alpha=0.5, ax=axes[0])
        axes[0].set_title("Song Distribution over Years")
        axes[0].set_xlabel("Release Date")
        axes[0].set_ylabel("Frequency")

        # Plot 2: Number of songs per decade
        sns.barplot(x=decade_counts.index, y=decade_counts.values, palette=sns.color_palette("Blues_r", len(decade_counts)), ax=axes[1])
        for p in axes[1].patches:
            axes[1].annotate(format(p.get_height(), '.0f'),
                             (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='center',
                             xytext=(0, 9),
                             textcoords='offset points')
        axes[1].set_title("Number of Songs per Decade")
        axes[1].set_xlabel("Decade")
        axes[1].set_ylabel("Number of Songs")
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()
    
    def genre_analysis(self) -> None:
        """Displays the distribution of genres and their evolution over decades."""
        genre_counts = self.df["tag"].value_counts()
        genre_distribution = self.df["tag"].value_counts(normalize=True)
        genre_year_counts = self.df.groupby(['decade', 'tag']).size().unstack(fill_value=0)

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        bars = genre_counts.plot(kind='bar', color=sns.color_palette("husl", len(genre_distribution)), ax=axes[0])
        axes[0].set_title('Genre Distribution')
        axes[0].set_xlabel('Genre')
        axes[0].set_ylabel('Number of Songs')
        for i, val in enumerate(genre_distribution):
            axes[0].text(i, genre_counts[i] + 10, f'{val * 100:.2f}%', ha='center', va='bottom')
        axes[0].tick_params(axis='x', rotation=0)

        sns.lineplot(data=genre_year_counts, dashes=False, ax=axes[1])
        axes[1].set_title('Genre Distribution Across Decades')
        axes[1].set_xlabel('Decade')
        axes[1].set_ylabel('Number of Songs')
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    file_path = "/Users/elenadelgovernatore/Desktop/GitHub/NLP/data/raw/raw_dataset.csv"
    full_df = pd.read_csv(file_path)
    eda = ExploratoryDataAnalysis(full_df)
    eda.overview()
    eda.variable_summary()
    eda.class_balance()
