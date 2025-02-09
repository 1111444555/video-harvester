from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit, QPushButton,
    QLabel, QProgressBar, QMessageBox, QLineEdit
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread
import yt_dlp
import whisper
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
import json
import sys
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import string

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

class ProcessorWorker(QObject):
    progress_signal = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(str)

    def __init__(self, model_name, video_urls, article_urls, output_file):
        super().__init__()
        self.model_name = model_name
        self.video_urls = video_urls
        self.article_urls = article_urls
        self.output_file = output_file

    def process(self):
        try:
            # Check for CUDA availability
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_signal.emit(f"Using {device.upper()} for Whisper model")

            # Load Whisper Model
            self.log_signal.emit(f"Loading Whisper model: {self.model_name}")
            model = whisper.load_model(self.model_name, device=device)

            # Load T5 model for tag generation
            self.log_signal.emit("Loading T5 model for tag generation...")
            tag_tokenizer = T5Tokenizer.from_pretrained("t5-small")
            tag_model = T5ForConditionalGeneration.from_pretrained("t5-small")
            tag_device = "cuda" if torch.cuda.is_available() else "cpu"
            self.log_signal.emit(f"Using {tag_device.upper()} for tag generation model")
            tag_model = tag_model.to(tag_device)


            def extract_keywords(text, num_keywords=10):
                """Extract keywords by identifying frequent nouns and meaningful phrases."""
                # Tokenization and lowercasing
                words = word_tokenize(text.lower())

                # Remove stopwords and punctuation
                stop_words = set(stopwords.words('english'))
                words = [word for word in words if word not in stop_words and word not in string.punctuation]

                # POS tagging to extract nouns
                tagged_words = nltk.pos_tag(words)
                nouns = [word for word, pos in tagged_words if pos in ['NN', 'NNS', 'NNP', 'NNPS']]

                # Count frequency of nouns
                noun_freq = Counter(nouns)

                # Extract key phrases using spaCy
                doc = nlp(text)
                key_phrases = [chunk.text for chunk in doc.noun_chunks]

                # Combine keywords and phrases
                combined_keywords = noun_freq.most_common(num_keywords) + key_phrases[:num_keywords]
                
                # Return unique keywords
                return list(set([kw[0] if isinstance(kw, tuple) else kw for kw in combined_keywords]))


            def generate_tags(headline, content):
                #input_text = f"Generate 15 comma-separated tags for: {headline}. {content[:5000]}"
                if len(content) > 10000:
                    input_text = f"Generate 15 comma-separated tags for: {headline}. {content[:5000]} ... {content[-5000:]}"
                else:
                    input_text = f"Generate 15 comma-separated tags for: {headline}. {content[:5000]}"
                input_ids = tag_tokenizer.encode(input_text, return_tensors="pt", 
                                                max_length=512, truncation=True).to(tag_device)
                outputs = tag_model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
                tags = tag_tokenizer.decode(outputs[0], skip_special_tokens=True)
                tag_string=', '.join(tags.split(', ')[:15])
                return ', '.join(extract_keywords(tag_string))

            # Prepare directories
            os.makedirs("audio_files", exist_ok=True)
            os.makedirs("cache", exist_ok=True)

            processed_urls_file = "cache/processed_urls.json"
            if os.path.exists(processed_urls_file):
                with open(processed_urls_file, "r") as f:
                    processed_urls = json.load(f)
            else:
                processed_urls = []

            results = []


            # Process videos
            def process_video(url):
                if url in processed_urls:
                    self.log_signal.emit(f"Skipping already processed video: {url}")
                    return
                try:
                    video_id = url.split("=")[-1]
                    audio_path = f"audio_files/{video_id}.wav"
                    ydl_opts = {
                        "format": "bestaudio/best",
                        "outtmpl": f"audio_files/{video_id}.%(ext)s",
                        "postprocessors": [{
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "wav",
                            "preferredquality": "192",
                        }],
                    }
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(url, download=True)
                        title = info_dict.get("title", "N/A")
                        description = info_dict.get("description", "N/A")
                    result = model.transcribe(audio_path)
                    transcript = result["text"]
                    results.append({
                        "URL": url,
                        "Title": title,
                        "Description": description,
                        "Content": transcript,
                    })
                    processed_urls.append(url)
                    self.log_signal.emit(f"Processed video: {title}")
                except Exception as e:
                    self.log_signal.emit(f"Error processing video {url}: {e}")

            def process_video(url):
                if url in processed_urls:
                    self.log_signal.emit(f"Skipping already processed video: {url}")
                    return
                try:
                    video_id = url.split("=")[-1]
                    audio_path = f"audio_files/{video_id}.wav"
                    
                    # Configure yt-dlp options
                    ydl_opts = {
                        "format": "bestaudio/best",
                        "outtmpl": f"audio_files/{video_id}.%(ext)s",
                        "postprocessors": [{
                            "key": "FFmpegExtractAudio",
                            "preferredcodec": "wav",
                            "preferredquality": "192",
                        }],
                        # Add a custom user agent
                        "http_headers": {
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                        },
                        # Add cookies file if available (optional)
                        "cookiefile": "cookies.txt",  # Path to your cookies file
                    }
                    
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info_dict = ydl.extract_info(url, download=True)
                        title = info_dict.get("title", "N/A")
                        description = info_dict.get("description", "N/A")
                    
                    result = model.transcribe(audio_path)
                    transcript = result["text"]
                    
                    tags = generate_tags(title, transcript)
                    
                    results.append({
                        "HEADLINE": title,
                        "WEBLINK": url,
                        "CONTENT": description,
                        "NOTES": "",
                        "TAG": tags
                    })
                    processed_urls.append(url)
                    self.log_signal.emit(f"Processed video: {title}")
                except Exception as e:
                    self.log_signal.emit(f"Error processing video {url}: {e}")

            # Process videos
            # def process_video(url):
            #     if url in processed_urls:
            #         self.log_signal.emit(f"Skipping already processed video: {url}")
            #         return
            #     try:
            #         video_id = url.split("=")[-1]
            #         audio_path = f"audio_files/{video_id}.wav"
            #         ydl_opts = {
            #             "format": "bestaudio/best",
            #             "outtmpl": f"audio_files/{video_id}.%(ext)s",
            #             "postprocessors": [{
            #                 "key": "FFmpegExtractAudio",
            #                 "preferredcodec": "wav",
            #                 "preferredquality": "192",
            #             }],
            #         }
            #         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            #             info_dict = ydl.extract_info(url, download=True)
            #             title = info_dict.get("title", "N/A")
            #             description = info_dict.get("description", "N/A")
                    
            #         result = model.transcribe(audio_path)
            #         transcript = result["text"]
                    
            #         tags = generate_tags(title, transcript)
                    
            #         results.append({
            #             "HEADLINE": title,
            #             "WEBLINK": url,
            #             "CONTENT": description,
            #             "NOTES": "",
            #             "TAG": tags
            #         })
            #         processed_urls.append(url)
            #         self.log_signal.emit(f"Processed video: {title}")
            #     except Exception as e:
            #         self.log_signal.emit(f"Error processing video {url}: {e}")

            # Process articles
            def process_article(url):
                if url in processed_urls:
                    self.log_signal.emit(f"Skipping already processed article: {url}")
                    return
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.content, "html.parser")
                    title = soup.title.string if soup.title else "N/A"
                    
                    # Get first paragraph and full content
                    paragraphs = soup.find_all("p")
                    first_paragraph = paragraphs[0].get_text() if paragraphs else "N/A"
                    full_content = " ".join(p.get_text() for p in paragraphs)
                    
                    tags = generate_tags(title, full_content)
                    
                    results.append({
                        "HEADLINE": title,
                        "WEBLINK": url,
                        "CONTENT": first_paragraph,
                        "NOTES": "",
                        "TAG": tags
                    })
                    processed_urls.append(url)
                    self.log_signal.emit(f"Processed article: {title}")
                except Exception as e:
                    self.log_signal.emit(f"Error processing article {url}: {e}")

            # Task execution
            total_tasks = len(self.video_urls) + len(self.article_urls)
            current_task = 0
            for video_url in self.video_urls:
                process_video(video_url)
                current_task += 1
                self.progress_signal.emit(int((current_task / total_tasks) * 100))

            for article_url in self.article_urls:
                process_article(article_url)
                current_task += 1
                self.progress_signal.emit(int((current_task / total_tasks) * 100))

            # Save results
            df = pd.DataFrame(results)
            df.to_excel(self.output_file, index=False, engine='openpyxl')
            with open(processed_urls_file, "w") as f:
                json.dump(processed_urls, f)
            self.finished_signal.emit(f"Content saved to {self.output_file}")
        except Exception as e:
            self.finished_signal.emit(f"An error occurred: {str(e)}")


class VideoArticleProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video & Article Processor")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # UI elements (same as before)
        self.layout.addWidget(QLabel("Whisper Model:"))
        self.model_input = QLineEdit("base")
        self.layout.addWidget(self.model_input)

        self.layout.addWidget(QLabel("Video URLs (one per line):"))
        self.video_urls_input = QTextEdit()
        self.layout.addWidget(self.video_urls_input)

        self.layout.addWidget(QLabel("Article URLs (one per line):"))
        self.article_urls_input = QTextEdit()
        self.layout.addWidget(self.article_urls_input)

        self.layout.addWidget(QLabel("Output File Name:"))
        self.output_path_input = QLineEdit("content.xlsx")
        self.layout.addWidget(self.output_path_input)

        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.layout.addWidget(self.start_button)

        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        self.layout.addWidget(QLabel("Logs:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.layout.addWidget(self.log_box)

    def log_message(self, message):
        self.log_box.append(message)
        self.log_box.ensureCursorVisible()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def processing_finished(self, message):
        QMessageBox.information(self, "Finished", message)

    def start_processing(self):
        model_name = self.model_input.text().strip()
        video_urls = [url.strip() for url in self.video_urls_input.toPlainText().split("\n") if url.strip()]
        article_urls = [url.strip() for url in self.article_urls_input.toPlainText().split("\n") if url.strip()]
        output_file = self.output_path_input.text().strip()

        self.worker = ProcessorWorker(model_name, video_urls, article_urls, output_file)
        self.worker_thread = QThread()

        self.worker.moveToThread(self.worker_thread)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.log_signal.connect(self.log_message)
        self.worker.finished_signal.connect(self.processing_finished)
        self.worker.finished_signal.connect(self.worker_thread.quit)

        self.worker_thread.started.connect(self.worker.process)
        self.worker_thread.start()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = VideoArticleProcessorApp()
    main_window.show()
    sys.exit(app.exec_())