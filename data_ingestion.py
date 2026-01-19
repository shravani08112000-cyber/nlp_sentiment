import pandas as pd
from datasets import load_dataset
from logger import get_logger

logger = get_logger(__name__)



def load_imdb_data(sample_size=None):

    """Load IMDB Movie dataset for sentiment Analysis"""

    # Load Dataset
    logger.info(f"Data Ingestion begining")
    dataset = load_dataset('imdb')
    logger.info(f"{dataset}")
    

    # Convert to pandas datframe
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])
    logger.info(f"These are trin and test df's {train_df.shape} and {test_df.shape}")

    # Map labels 0=Negative and 1=positive We create a column called sentiment and map
    train_df['sentiment'] = train_df['label'].map({0: 'negative', 1: 'positive'})
    test_df['sentiment'] = test_df['label'].map({0: 'negative', 1: 'positive'})

    # Rename columns
    train_df = train_df.rename(columns={'text':'review'})
    test_df = test_df.rename(columns={'text':'review'})

    # sample data based on requested
    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

    print(f" Loaded {len(train_df)} training sample")
    print(f" Loaded {len(test_df)} testingsample")

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = load_imdb_data(sample_size=2000)
    print(train_df.head())
    print(test_df.head())
    logger.info(f"Data {train_df.head()}")