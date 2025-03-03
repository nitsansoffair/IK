import matplotlib
import speech_recognition as sr
import os
import torch
import networkx as nx
import matplotlib.pyplot as plt

from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use('Agg')  # Use a backend that doesn't require a GUI

# Function to extract audio from video
def extract_audio_from_video(video_path, audio_path):
    # Use ffmpeg to extract audio from the video
    video = AudioSegment.from_file(video_path)
    audio = video.split_to_mono()[0]  # Split stereo if needed, here we assume mono is fine
    audio.export(audio_path, format="wav")

# Function to transcribe Hebrew speech to text using SpeechRecognition
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)  # Capture the audio from file

    try:
        # Use Google Web Speech API (can be used for Hebrew as well)
        text = recognizer.recognize_google(audio, language="he-IL")
        return text
    except sr.UnknownValueError:
        return "Audio not understood"
    except sr.RequestError as e:
        return f"Error with the speech recognition service: {e}"

# Function to extract text embeddings using BERT (for Hebrew text)
def extract_text_embedding(bert_model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

# Function to compute the similarity between two embeddings (cosine similarity)
def compute_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1, embedding2)[0][0]

# Create the graph based on similarities
def create_video_graph(video_files, text_data):
    G = nx.Graph()

    # Add nodes
    for video_file in video_files:
        G.add_node(video_file)

    # Add edges based on similarity (use text, audio, and image embeddings)
    for i, video1 in enumerate(video_files):
        print(f"Similarity Estimation: {i}/{len(video_files)}")

        for j, video2 in enumerate(video_files):
            if i < j:  # Only compute for pairs
                text_sim = compute_similarity(text_data[video1], text_data[video2])

                # Calculate a combined similarity score (weighted)
                combined_similarity = text_sim

                if combined_similarity > 0.5:  # Only add edges for a significant similarity
                    G.add_edge(video1, video2, weight=combined_similarity)

    return G

# Function to visualize the graph
def visualize_graph(G, category):
    pos = nx.spring_layout(G)  # Layout for visualization
    weights = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=weights)

    plt.title("Video Similarity Graph")
    plt.axis("off")

    plt.savefig(f"./results/{category}.png")

# Analyze all videos in a category folder and compute embeddings
def analyze_videos_and_create_graph(category_folder, bert_model, tokenizer):
    video_files = [f for f in os.listdir(category_folder) if f.endswith(".mp4")]
    text_data = {}

    index = 0

    for video_file in video_files:
        index += 1
        print(f"Data Processing: {index}/{len(video_files)}")

        video_path = os.path.join(category_folder, video_file)

        # Extract audio
        audio_path = video_path.replace(".mp4", ".wav")
        extract_audio_from_video(video_path, audio_path)

        # Extract text (from the video transcription, can use pre-existing code)
        transcription = transcribe_audio(video_path.replace(".mp4", ".wav"))  # assuming audio extraction is done
        text_data[video_file] = extract_text_embedding(bert_model, tokenizer, transcription)

    # Create the graph based on the embeddings
    G = create_video_graph(video_files, text_data)

    category = category_folder.split('/')[-1]

    # Visualize the graph
    visualize_graph(G, category)