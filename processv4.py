import tkinter as tk
from tkinter import ttk, messagebox
import yt_dlp
import whisper
import pandas as pd
import json
import os
import requests
from bs4 import BeautifulSoup
import time
import threading

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Video & Article Processor")
        self.root.geometry("800x600")

        # Variables for user inputs
        self.whisper_model = tk.StringVar(value="base")
        self.video_urls = tk.StringVar()
        self.article_urls = tk.StringVar()
        self.output_file_path = tk.StringVar(value="content.csv")
        self.progress = tk.DoubleVar()
        self.log_text = tk.StringVar(value="Ready to start...\n")

        # Build GUI
        self.create_widgets()

    def create_widgets(self):
        # Input for Whisper Model
        tk.Label(self.root, text="Whisper Model:").pack(anchor="w", padx=20, pady=5)
        tk.Entry(self.root, textvariable=self.whisper_model, width=50).pack(anchor="w", padx=20)

        # Input for Video URLs
        tk.Label(self.root, text="Video URLs (comma-separated):").pack(anchor="w", padx=20, pady=5)
        tk.Entry(self.root, textvariable=self.video_urls, width=80).pack(anchor="w", padx=20)

        # Input for Article URLs
        tk.Label(self.root, text="Article URLs (comma-separated):").pack(anchor="w", padx=20, pady=5)
        tk.Entry(self.root, textvariable=self.article_urls, width=80).pack(anchor="w", padx=20)

        # Input for Output File Path
        tk.Label(self.root, text="Output File Name:").pack(anchor="w", padx=20, pady=5)
        tk.Entry(self.root, textvariable=self.output_file_path, width=50).pack(anchor="w", padx=20)

        # Start Processing Button
        tk.Button(self.root, text="Start Processing", command=self.start_processing).pack(pady=20)

        # Progress Bar
        ttk.Progressbar(self.root, variable=self.progress, maximum=100).pack(pady=10, fill="x", padx=20)

        # Logs Box
        tk.Label(self.root, text="Logs:").pack(anchor="w", padx=20)
        self.log_box = tk.Text(self.root, height=15, wrap="word", state="normal")
        self.log_box.pack(padx=20, pady=10, fill="both", expand=True)

        # Output Path Display
        tk.Label(self.root, text="Processed Output File:").pack(anchor="w", padx=20, pady=5)
        tk.Entry(self.root, textvariable=self.output_file_path, state="readonly", width=50).pack(anchor="w", padx=20)

    def log_message(self, message):
        self.log_box.config(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def update_progress(self, value):
        self.progress.set(value)
        self.root.update_idletasks()

    def process(self):
        try:
            # Get user inputs
            model_name = self.whisper_model.get()
            video_urls = [url.strip() for url in self.video_urls.get().split(",") if url.strip()]
            article_urls = [url.strip() for url in self.article_urls.get().split(",") if url.strip()]
            output_file = self.output_file_path.get()

            if not video_urls and not article_urls:
                raise ValueError("No URLs provided for processing.")

            # Load Whisper Model
            self.log_message(f"Loading Whisper model: {model_name}")
            model = whisper.load_model(model_name)

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
                    self.log_message(f"Skipping already processed video: {url}")
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
                    self.log_message(f"Processed video: {title}")
                except Exception as e:
                    self.log_message(f"Error processing video {url}: {e}")

            # Process articles
            def process_article(url):
                if url in processed_urls:
                    self.log_message(f"Skipping already processed article: {url}")
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
                    self.log_message(f"Processed article: {title}")
                except Exception as e:
                    self.log_message(f"Error processing article {url}: {e}")

            # Task execution
            total_tasks = len(video_urls) + len(article_urls)
            current_task = 0
            for video_url in video_urls:
                process_video(video_url)
                current_task += 1
                self.update_progress((current_task / total_tasks) * 100)

            for article_url in article_urls:
                process_article(article_url)
                current_task += 1
                self.update_progress((current_task / total_tasks) * 100)

            # Save results
            df = pd.DataFrame(results)
            df.to_csv(output_file, index=False)
            with open(processed_urls_file, "w") as f:
                json.dump(processed_urls, f)
            self.log_message(f"Content saved to {output_file}")
            messagebox.showinfo("Success", "Processing completed successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def start_processing(self):
        threading.Thread(target=self.process).start()

# Run the Application
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()