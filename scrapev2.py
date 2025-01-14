import yt_dlp
from pydub import AudioSegment
import whisper
import pandas as pd
import json
import os

# Load configuration
with open('./config.json', 'r') as f:
    config = json.load(f)

# Initialize Whisper model
model = whisper.load_model("tiny")

# Prepare list to store results
results = []

# Create a directory to store audio files
os.makedirs('audio_files', exist_ok=True)

# Function to download audio and transcribe
def process_video(url):
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
            'Transcript': transcript
        })

        print(f'Processed: {title}')
    except Exception as e:
        print(f'Error processing {url}: {e}')

# Process each video URL
for video_url in config['videos']:
    process_video(video_url)

# Save results to CSV
df = pd.DataFrame(results)
df.to_csv('transcriptions.csv', index=False)

print('Transcriptions saved to transcriptions.csv')
