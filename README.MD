# MusicKeyAndBPM
A python based audio tempo and key identifier which can be run through a browser. It can be hosted using fastAPI and uvicorn based server. 


## Installation 
```bash
git clone https://github.com/jashjasani/MusicKeyAndBPM
cd MusicKeyAndBPM
pip install -r requirements.txt
```

##Dependencies 
Linux
```bash
sudo apt update
sudo apt install ffmpeg
```
For windows ffmpeg can be downloaded from :  https://ffmpeg.org/download.html

## Running the server 
```bash
uvicorn main:app --reload
```

This will start server on port 8000
