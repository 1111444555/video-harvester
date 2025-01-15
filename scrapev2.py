import yt_dlp
import whisper
import pandas as pd
import json
import os
import requests
from bs4 import BeautifulSoup
import time

# Load configuration
with open('./config.json', 'r') as f:
    config = json.load(f)

# Initialize Whisper model
model_name=config["whisper_model"]
model = whisper.load_model(model_name)

# Prepare list to store results
results = []

# Create directories to store audio files and cache
os.makedirs('audio_files', exist_ok=True)
os.makedirs('cache', exist_ok=True)

# Load processed URLs from cache
processed_urls_file = 'cache/processed_urls.json'
if os.path.exists(processed_urls_file):
    with open(processed_urls_file, 'r') as f:
        processed_urls = json.load(f)
else:
    processed_urls = []

# Function to download audio and transcribe
def process_video(url):
    if url in processed_urls:
        print(f'Skipping already processed video: {url}')
        return
    try:
        # Extract video ID from URL
        video_id = url.split('=')[-1]
        audio_path = f'audio_files/{video_id}.wav'

        # Download audio using yt-dlp
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'audio_files/{video_id}.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            title = info_dict.get('title', 'N/A')
            description = info_dict.get('description', 'N/A')

        # Transcribe audio using Whisper
        result = model.transcribe(audio_path)
        transcript = result['text']

        # Append results
        results.append({
            'URL': url,
            'Title': title,
            'Description': description,
            'Content': transcript
        })

        # Mark URL as processed
        processed_urls.append(url)
        print(f'Processed video: {title}')
    except Exception as e:
        print(f'Error processing video {url}: {e}')

# Function to extract article headline and content
def process_article(url):
    if url in processed_urls:
        print(f'Skipping already processed article: {url}')
        return
    try:

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:104.0) Gecko/20100101 Firefox/104.0',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        response = requests.get(url, headers=headers)
        #response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract title and content
        title = soup.title.string if soup.title else 'N/A'
        content = ' '.join(p.get_text() for p in soup.find_all('p'))

        # Append results
        results.append({
            'URL': url,
            'Title': title,
            'Description': 'N/A',
            'Content': content
        })

        # Mark URL as processed
        processed_urls.append(url)
        print(f'Processed article: {title}')
    except Exception as e:
        print(f'Error processing article {url}: {e}')

# Process each video URL
for video_url in config.get('videos', []):
    process_video(video_url)
    time.sleep(2)

# Process each article URL
for article_url in config.get('articles', []):
    process_article(article_url)
    time.sleep(2)


# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('content.csv', index=False)

# Update processed URLs cache
with open(processed_urls_file, 'w') as f:
    json.dump(processed_urls, f)

print('Content saved to content.csv')
