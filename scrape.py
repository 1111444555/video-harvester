#from pytube import YouTube

# Replace with your YouTube video URL
video_id = 'https://www.youtube.com/watch?v=soVSPe3OBKU'

# # Create a YouTube object
# yt = YouTube(video_url)

# # Extract title and description
# title = yt.title
# description = yt.description

# print(f'Title: {title}')
# print(f'Description: {description}')


from youtube_transcript_api import YouTubeTranscriptApi

# Replace with your YouTube video ID
video_id = 'your_video_id'

# Fetch the transcript
transcript = YouTubeTranscriptApi.get_transcript(video_id)

# Combine transcript text
transcript_text = ' '.join([entry['text'] for entry in transcript])

print(f'Transcript: {transcript_text}')
