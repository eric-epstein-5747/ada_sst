import argparse
import ssl
import math
import os
import subprocess
from typing import Tuple
from tqdm import tqdm
from faster_whisper import WhisperModel
import pandas as pd

model_size = "large-v2"

ssl._create_default_https_context = ssl._create_unverified_context

# Constants
max_bytes = 26214400  # From Whisper error message
overlap_seconds = 0


def get_chunkdur_num_chunks(
    filename: str,
) -> Tuple[float, int]:
    # mandating 300 second chunks
    chunk_duration_s = 10

    # Get the duration of the audio file
    audio_duration_s = float(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filename,
            ]
        ).strip()
    )

    # Calculate the number of chunks
    num_chunks = math.ceil(audio_duration_s / (chunk_duration_s - overlap_seconds))

    return chunk_duration_s, num_chunks


def main(filename: str, qualitative_name: str):
    print("Loading model")
    # run on CPU with INT8
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    print("Getting chunk duration & number of chunks")
    chunk_duration_s, num_chunks = get_chunkdur_num_chunks(filename)

    output_folder = "./data/faster_whisper_chunks"
    os.makedirs(output_folder, exist_ok=True)

    # Get the file extension from the filename
    file_extension = os.path.splitext(filename)[1]

    # initialize dict for storing word timestamps
    word_dict = {
        "word": [],
        "word_start": [],
        "word_end": [],
    }

    print("Using ffmpeg to extract & transcribe each chunk...")
    for i in tqdm(range(num_chunks)):
        start_s = i * (chunk_duration_s - overlap_seconds)
        end_s = start_s + chunk_duration_s

        # Save the chunk to disk
        chunk_file = os.path.join(output_folder, f"chunk_{i + 1}{file_extension}")

        # Use ffmpeg to extract the chunk directly into the compressed format (m4a)
        subprocess.call(
            [
                "ffmpeg",
                "-ss",
                str(start_s),
                "-i",
                filename,
                "-t",
                str(chunk_duration_s),
                "-vn",
                "-acodec",
                "copy",
                "-y",
                chunk_file,
            ]
        )

        ## Transcribe the chunk
        segments, _ = model.transcribe(chunk_file, word_timestamps=True)
        for segment in segments:
            for word in segment.words:
                word_dict["word"].append(word.word)
                word_dict["word_start"].append(start_s + word.start)
                word_dict["word_end"].append(start_s + word.end)

    # Save transcriptions to a file
    print(
        f"Saving transcriptions to ./results/non_diarized_transcriptions/{qualitative_name}_transcriptions.csv"
    )
    os.makedirs("./results/non_diarized_transcriptions/", exist_ok=True)
    df = pd.DataFrame(word_dict)
    df.to_csv(
        f"./results/non_diarized_transcriptions/{qualitative_name}_transcriptions.csv",
        header=True,
        index=False,
    )
    return


def parse_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_audio_file",
        required=False,
        default="./data/original_files/david_and_ada_side_1_short.wav",
        type=str,
        help="Path to .wav file of the audio you want to transcribe",
    )
    parser.add_argument(
        "--qualitative_name",
        required=False,
        default="david_and_ada_side_1_short",
        type=str,
        help="Qualitative description of your intended output file (no spaces, dots, or capital letters!)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argv()

    main(
        filename=args.input_audio_file,
        qualitative_name=args.qualitative_name,
    )
