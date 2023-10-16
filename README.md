# Ada Speech-to-Text

The goal of this project is to create a speech-to-text tool that can be easily used to transcribe digitized tape-recorded interviews. It was inspired by wanting to keep alive the memory of my grandmother.

# Setup

## On MacOS

1. Clone this repo using `git clone`
2. Create a project environment: `python3 -m venv .venv`
3. Activate it: `source .venv/bin/activate`
4. Install all required packages: `pip install -r requirements.txt`
5. Create a data directory and add your files:
   - `mkdir src/data`
   - `mkdir src/data/original_files`
   - `mv [PATH_TO_YOUR_FILE] src/data/original_files`
6. Run the main script (from within the `src` directory): `python3 run_whisper_large.py --input_audio_file data/original_files/YOUR_FILE_NAME`
7. Transcription results will appear in `src/results/non_diarized/transcriptions.txt`

# TO DO

1. Use diarization file from `data/diarization/diarization.txt` to diarize `transcriptions.txt` above; or simply get `diarize_and_transcribe.py` to produce more accurate transcriptions.
2. Improve the accuracy of the diarization in `diarization.txt`.
