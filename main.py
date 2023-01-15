# created by jashjasani : https://github.com/jashjasani
import io

import soundfile
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import os
from fastapi.staticfiles import StaticFiles
import librosa
import librosa.effects
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import aiofiles
from fastapi.responses import HTMLResponse


class Tonal_Fragment(object):
    def _init_(self, waveform, sr, tstart=None, tend=None):
        self.waveform = waveform
        self.sr = sr
        self.tstart = tstart
        self.tend = tend

        if self.tstart is not None:
            self.tstart = librosa.time_to_samples(self.tstart, sr=self.sr)
        if self.tend is not None:
            self.tend = librosa.time_to_samples(self.tend, sr=self.sr)
        self.y_segment = self.waveform[self.tstart:self.tend]
        self.chromograph = librosa.feature.chroma_cqt(y=self.y_segment, sr=self.sr, bins_per_octave=24)

        # chroma_vals is the amount of each pitch class present in this time interval
        self.chroma_vals = []
        for i in range(12):
            self.chroma_vals.append(np.sum(self.chromograph[i]))
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # dictionary relating pitch names to the associated intensity in the song
        self.keyfreqs = {pitches[i]: self.chroma_vals[i] for i in range(12)}

        keys = [pitches[i] + ' major' for i in range(12)] + [pitches[i] + ' minor' for i in range(12)]

        # use of the Krumhansl-Schmuckler key-finding algorithm, which compares the chroma
        # data above to typical profiles of major and minor keys:
        maj_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        min_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        # finds correlations between the amount of each pitch class in the time interval and the above profiles,
        # starting on each of the 12 pitches. then creates dict of the musical keys (major/minor) to the correlation
        self.min_key_corrs = []
        self.maj_key_corrs = []
        for i in range(12):
            key_test = [self.keyfreqs.get(pitches[(i + m) % 12]) for m in range(12)]
            # correlation coefficients (strengths of correlation for each key)
            self.maj_key_corrs.append(round(np.corrcoef(maj_profile, key_test)[1, 0], 3))
            self.min_key_corrs.append(round(np.corrcoef(min_profile, key_test)[1, 0], 3))

        # names of all major and minor keys
        self.key_dict = {**{keys[i]: self.maj_key_corrs[i] for i in range(12)},
                         **{keys[i + 12]: self.min_key_corrs[i] for i in range(12)}}

        # this attribute represents the key determined by the algorithm
        self.key = max(self.key_dict, key=self.key_dict.get)
        self.bestcorr = max(self.key_dict.values())

        # this attribute represents the second-best key determined by the algorithm,
        # if the correlation is close to that of the actual key determined
        self.altkey = None
        self.altbestcorr = None

        for key, corr in self.key_dict.items():
            if corr > self.bestcorr * 0.9 and corr != self.bestcorr:
                self.altkey = key
                self.altbestcorr = corr

    # prints the relative prominence of each pitch class
    def print_chroma(self):
        self.chroma_max = max(self.chroma_vals)
        for key, chrom in self.keyfreqs.items():
            print(key, '\t', f'{chrom / self.chroma_max:5.3f}')

    # prints the correlation coefficients associated with each major/minor key
    def corr_table(self):
        for key, corr in self.key_dict.items():
            print(key, '\t', f'{corr:6.3f}')

    # printout of the key determined by the algorithm; if another key is close, that key is mentioned
    def findKey(self):
        return max(self.key_dict, key=self.key_dict.get)
        # print("likely key: ", max(self.key_dict, key=self.key_dict.get), ", correlation: ", self.bestcorr, sep='')
        # if self.altkey is not None:
        #     print("also possible: ", self.altkey, ", correlation: ", self.altbestcorr, sep='')

    # prints a chromagram of the file, showing the intensity of each pitch class over time
    def chromagram(self, title=None):
        C = librosa.feature.chroma_cqt(y=self.waveform, sr=self.sr, bins_per_octave=24)
        plt.figure(figsize=(12, 4))
        img = io.BytesIO
        librosa.display.specshow(C, sr=self.sr, x_axis='time', y_axis='chroma', vmin=0, vmax=1)
        if title is None:
            plt.title('Chromagram')
        else:
            plt.title(title)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(img,format='png')
        return img


def libLoader(filename):
    y, sr = librosa.load(filename)
    y_harmonic, y_percussive = librosa.effects.hpss
    return y, sr, y_harmonic, y_percussive


def bpmAndKeyFinder(filename):
    y, sr, y_harmonic, y_percussive = libLoader(filename)

    tempo, beat = librosa.beat.beat_track(y=y, sr=sr)

    Tone = Tonal_Fragment(y_harmonic, sr)

    key = Tone.findKey()

    os.remove(filename)

    return tempo, key


def generate_html():
    html_response = open('pages/index.html', 'r')
    html = html_response.read()
    return HTMLResponse(content=html, status_code=200)


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory='static'), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    return generate_html()


@app.post("/")
async def root(request: Request):
    form = await request.form()
    file = form["file"]
    audio_format = file.content_type
    ##audio_file = file.file
    async with aiofiles.open('files/audio.'+audio_format.split('/')[-1], 'wb') as out_file:
        content = await file.read()  # async read
        await out_file.write(content)  # async write
    bpm, key = bpmAndKeyFinder('files/audio.'+audio_format.split('/')[-1])
    return {
        "bpm": bpm,
        "key": key
    }