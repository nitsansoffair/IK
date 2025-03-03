import argparse
import zipfile
import torch

from transformers import BertModel, BertTokenizer
from G import analyze_videos_and_create_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_zip():
    # Extract the ZIP file
    zip_path = "./datasets/IK.zip"
    extract_path = "./datasets"

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_type", default="text", type=str)
    parser.add_argument("--category_type", default="idialists", type=str)
    args = parser.parse_args()

    print(f"Category Type: {args.category_type}, Data Type: {args.data_type}")

    # Initialize BERT model and tokenizer for text embeddings (Hebrew)
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

    # Example: Analyze videos in the 'idialists' category and display the graph
    category_folder = f"./datasets/{args.category_type}"

    analyze_videos_and_create_graph(args, category_folder, bert_model, tokenizer)