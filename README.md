
# Steps

- Modify config.json
- For Whisper Speech-to-text model, specify type of model to use, refer table at https://github.com/openai/whisper. 

```
pip install -r requirements.txt
python3 -c "import nltk ; nltk.download('punkt_tab')  ; nltk.download('averaged_perceptron_tagger_eng')" 
python3 scrapev3.py 

If you are using Windows, run following command(s) in terminal(only one time) to build .exe

pip3 install pyinstaller
pyinstaller --onefile --noconsole scrapev6.py

```

- output should be content.csv in the same folder

# Notes
- Requires GPU to run Speech to text transcription with Whisper AI model, otherwise it will default to cpu and its very slow.
