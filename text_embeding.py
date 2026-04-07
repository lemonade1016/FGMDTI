import os
import pandas as pd
from openai import OpenAI
import numpy as np
from retry import retry

# Initialize OpenAI client
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY", "sk-c6cadacbab8b46a9ab57bf97a0f022f8"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

@retry(tries=3, delay=1, backoff=2)
def get_embedding(text, model="text-embedding-v4", dimensions=1024):
    """
    Generate text embedding using the text-embedding-v4 model.

    Parameters:
        text (str): Input text
        model (str): Model name, default is text-embedding-v4
        dimensions (int): Embedding vector dimension, default is 1024

    Returns:
        list: Embedding vector (list of floats), or None if failed
    """
    try:
        if not isinstance(text, str) or not text.strip():
            print(f"Warning: Invalid text, skipping embedding: {text}")
            return None
        response = client.embeddings.create(
            model=model,
            input=text.strip(),
            dimensions=dimensions,
            encoding_format="float"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error: Unable to generate embedding for text: {text[:50]}...: {e}")
        return None

def process_embeddings(input_csv, output_npy):
    """
    Read CSV file, generate embeddings for 'domain_text_desc' column, and save to .npy file.

    Parameters:
        input_csv (str): Input CSV file path
        output_npy (str): Output .npy file path
    """
    # Read CSV file, skipping the header
    try:
        df = pd.read_csv(input_csv, skiprows=1, header=None,
                        names=['domain_id', 'domain_name', 'domain_seq', 'domain_text_desc'])
    except Exception as e:
        print(f"Error: Unable to read CSV file {input_csv}: {e}")
        return

    # Verify the required column
    if 'domain_text_desc' not in df.columns:
        print("Error: CSV file is missing the 'domain_text_desc' column.")
        return

    # Generate embeddings
    embeddings = []
    failed_count = 0
    print("Generating embeddings for 'domain_text_desc'...")
    for idx, text in enumerate(df['domain_text_desc']):
        embedding = get_embedding(text)
        if embedding is not None:
            embeddings.append(embedding)
        else:
            # Use zero vector if embedding fails
            embeddings.append([0] * 1024)
            failed_count += 1
            print(f"Warning: Embedding generation failed for row {idx + 1} (0-based index). Using zero vector.")

    # Convert to NumPy array
    embeddings_array = np.array(embeddings, dtype=np.float32)

    # Save to .npy file
    try:
        np.save(output_npy, embeddings_array)
        print(f"Embeddings saved to {output_npy}, shape: {embeddings_array.shape}")
    except Exception as e:
        print(f"Error: Unable to save .npy file {output_npy}: {e}")

    # Report failures
    if failed_count > 0:
        print(f"Warning: {failed_count} embeddings failed to generate and were replaced with zero vectors.")

if __name__ == "__main__":
    # Input and output file paths
    input_csv = "/media/4T2/lmd/Qwen2.5/drug/protein_analysis/data_process/domain_ew/biosnap/biosnap_domain_mapping.csv"
    output_npy = "/media/4T2/lmd/Qwen2.5/drug/protein_analysis/data_process/domain_ew/biosnap/biosnap_domain_text.npy"

    # Process embeddings
    process_embeddings(input_csv, output_npy)