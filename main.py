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
    # Paths to video folders
    video_path = "/content/videos"
    categories = ["idialists", "politicians"]

    # Initialize BERT model and tokenizer for text embeddings (Hebrew)
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    bert_model = BertModel.from_pretrained("bert-base-multilingual-cased")

    # Example: Analyze videos in the 'idialists' category and display the graph
    category_folder = "./datasets/idialists"
    analyze_videos_and_create_graph(category_folder, bert_model, tokenizer)