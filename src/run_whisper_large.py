import argparse
import ssl
import whisper
import math
import os
import subprocess
from typing import Tuple

ssl._create_default_https_context = ssl._create_unverified_context

# Constants
max_bytes = 26214400  # From Whisper error message
overlap_seconds = 5


def get_chunkdur_num_chunks(
    filename: str,
) -> Tuple[float, int]:
    # # Get the bit rate directly from the file
    # bit_rate = float(
    #     subprocess.check_output(
    #         [
    #             "ffprobe",
    #             "-v",
    #             "quiet",
    #             "-show_entries",
    #             "format=bit_rate",
    #             "-of",
    #             "default=noprint_wrappers=1:nokey=1",
    #             filename,
    #         ]
    #     ).strip()
    # )

    # # Estimate the duration of each chunk
    # chunk_duration_s = (max_bytes * 8.0) / bit_rate * 0.9
    # mandating 30 second chunks
    chunk_duration_s = 30

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


def main(filename):
    model = whisper.load_model("large")

    chunk_duration_s, num_chunks = get_chunkdur_num_chunks(filename)

    transcriptions = []

    output_folder = "./data/chunks"
    os.makedirs(output_folder, exist_ok=True)

    # Get the file extension from the filename
    file_extension = os.path.splitext(filename)[1]

    for i in range(num_chunks):
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
        # load audio and pad/trim it to fit 30 seconds
        audio = whisper.load_audio(chunk_file)
        audio = whisper.pad_or_trim(audio)
        # make log-Mel spectrogram and move to the same device as the model
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        # decode the audio
        options = whisper.DecodingOptions(fp16=False)
        transcription = whisper.decode(model, mel, options).text
        transcriptions.append(transcription)

    # Save transcriptions to a file
    os.makedirs("./results/non_diarized/", exist_ok=True)
    with open("./results/non_diarized/transcriptions.txt", "w") as file:
        for idx, transcription in enumerate(transcriptions):
            file.write(f"Chunk {idx + 1}:\n{transcription}\n\n")

    return


def parse_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_audio_file",
        required=False,
        default="./data/david_and_ada_side_1_short.wav",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argv()

    filename = args.input_audio_file

    main(filename)
