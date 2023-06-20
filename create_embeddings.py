import pandas as pd
import openai
from openai.embeddings_utils import get_embedding
import pinecone
import numpy as np
import itertools
from tqdm import tqdm

# Create the parser
parser = argparse.ArgumentParser(description='Load a CSV file into a pandas DataFrame.')

# Add an argument
parser.add_argument('file_path', type=str, help='The path to the CSV file.')

# Parse the command line arguments
args = parser.parse_args()

# Load the CSV file
df = pd.read_csv(args.file_path, low_memory=False)

# Initialize OpenAI and Pinecone
openai.api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone.init(api_key=pinecone_api_key, environment='eu-west1-gcp')

# Create a Pinecone index
index = pinecone.Index("bauhaus-stg-products-paid")

# Define a helper function to split data into chunks
def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

# Calculate total number of batches
total_batches = len(df) // 100 + (len(df) % 100 > 0)

# Iterate over the products in batches
for batch in tqdm(chunks(df.iterrows()), total=total_batches, desc="Processing batches"):
    batch_data = []
    for idx, row in batch:
        # Preprocess the data, include column names with the values
        text = ' '.join([f'{col}: {val}' for col, val in row.astype(str).to_dict().items()])

        # Create the embeddings with text-embedding-ada-002
        embeddings = get_embedding(text, engine='text-embedding-ada-002')

        # Convert the list of embeddings to a numpy array
        embeddings_array = np.array(embeddings).flatten()

       # Add the data to the batch
        metadata = {'name': row['name'], 'description': row['description'], 'base_image': row['base_image']}
        batch_data.append((row['url_key'], embeddings_array.tolist(), metadata))

    # Upsert the batch to the Pinecone index
    index.upsert(batch_data)

