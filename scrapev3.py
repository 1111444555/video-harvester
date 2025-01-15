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
            # Load Whisper Model
            self.log_signal.emit(f"Loading Whisper model: {self.model_name}")
            model = whisper.load_model(self.model_name)

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
                    content = " ".join(p.get_text() for p in soup.find_all("p"))
                    results.append({
                        "URL": url,
                        "Title": title,
                        "Description": "N/A",
                        "Content": content,
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
                self.progress_signal.emit((current_task / total_tasks) * 100)

            for article_url in self.article_urls:
                process_article(article_url)
                current_task += 1
                self.progress_signal.emit((current_task / total_tasks) * 100)

            # Save results
            df = pd.DataFrame(results)
            df.to_csv(self.output_file, index=False)
            with open(processed_urls_file, "w") as f:
                json.dump(processed_urls, f)
            self.finished_signal.emit(f"Content saved to {self.output_file}")
        except Exception as e:
            self.finished_signal.emit(f"An error occurred: {e}")


class VideoArticleProcessorApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Video & Article Processor")
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # Whisper Model Input
        self.layout.addWidget(QLabel("Whisper Model:"))
        self.model_input = QLineEdit("base")
        self.layout.addWidget(self.model_input)

        # Video URLs Input
        self.layout.addWidget(QLabel("Video URLs (one per line):"))
        self.video_urls_input = QTextEdit()
        self.layout.addWidget(self.video_urls_input)

        # Article URLs Input
        self.layout.addWidget(QLabel("Article URLs (one per line):"))
        self.article_urls_input = QTextEdit()
        self.layout.addWidget(self.article_urls_input)

        # Output File Path Input
        self.layout.addWidget(QLabel("Output File Name:"))
        self.output_path_input = QLineEdit("content.csv")
        self.layout.addWidget(self.output_path_input)

        # Start Processing Button
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        self.layout.addWidget(self.start_button)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.layout.addWidget(self.progress_bar)

        # Logs Display
        self.layout.addWidget(QLabel("Logs:"))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.layout.addWidget(self.log_box)

    def log_message(self, message):
        """Append a log message to the logs display."""
        self.log_box.append(message)
        self.log_box.ensureCursorVisible()

    def update_progress(self, value):
        """Update progress bar."""
        self.progress_bar.setValue(value)

    def processing_finished(self, message):
        """Handle the completion of the processing."""
        QMessageBox.information(self, "Finished", message)

    def start_processing(self):
        """Start processing in a separate thread."""
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
