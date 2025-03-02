# Description: This script is used to split the text into sentences using nltk library.
# The text is a string containing multiple documents. Each document is separated by a newline character.
# The script uses the sent_tokenize function from the nltk.tokenize module to split the text into sentences.
import nltk
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize
import re
import os

# TODO: Add a function to load the documents from the PDF files.
# def load_documents_from_pdf():
#     pass


def load_documents(data_dir: str) -> list[str]:
    """
    Load and split documents from text files in the given directory.
    Assumes each document starts with a line like "###<number>".
    Args:
        data_dir (str): The directory containing the text files.
    Returns:
        list[str]: A list of documents.
    1. The function first checks if the directory exists.
    2. It then lists all text files in the directory.
    3. For each file, it reads the content and splits it into documents.
    4. Each document is assumed to start with a line like "###<number>".
    5. The documents are cleaned and stored in a list.
    6. Finally, the list of documents is returned.  
    """
   
    documents = []
    file_patterns = [file for file in os.listdir(data_dir) if file.endswith(".txt")]

    if not file_patterns:
        print("No text files found in the given directory.")
        return

    for pattern in file_patterns:
        file_path = os.path.join(data_dir, pattern)
        if os.path.exists(file_path):
            print(f"Loading documents from {pattern}...")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            # Split the file into records based on lines starting with "###" and digits.
            records = re.split(r'(?=^###\d+)', text, flags=re.MULTILINE)
            # Clean and filter out any empty records.
            records = [re.sub(r'^###\d+\s*', '', record.strip()) for record in records if record.strip()]
            print(f"Found {len(records)} records in {pattern}.")
            documents.extend(records)
        else:
            print(f"Warning: {file_path} not found.")
        
    return documents

def main():
    # Load and split documents from the text files.
    documents = load_documents()
    if not documents:
        raise ValueError("No documents found. Please check your dataset files.")
    
    print(f"Total documents loaded: {len(documents)}")

    # Double check how many words are in each document in average and maximum number of words in a document
    len_total_words = [len(doc.split()) for doc in documents]
    avg_words = sum(len_total_words) / len(documents)
    min_words = min(len_total_words)
    max_words = max(len_total_words)
    print(f"Average number of words in a document: {avg_words:.2f}")
    print(f"Minimum number of words in a document: {min_words}")
    print(f"Maximum number of words in a document: {max_words}")

    # for doc in documents[0:5]:
    #     print(f"Document: {doc}\n")
    #     print('-' * 50)

if __name__ == "__main__":
    main()

