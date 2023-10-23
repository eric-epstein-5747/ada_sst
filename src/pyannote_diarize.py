import argparse
from pyannote.audio import Pipeline
from pydub import AudioSegment
import torch
import re
import os
from mysecrets import hf_token


def millisec(timeStr):
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
    return s


def group_and_save(
    diarization_file: str,
    prepended_input_file: str,
) -> None:
    dzs = open(diarization_file).read().splitlines()
    groups = []
    g = []
    lastend = 0

    for d in dzs:
        if g and (g[0].split()[-1] != d.split()[-1]):  # same speaker
            groups.append(g)
            g = []
        g.append(d)

        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=d)[1]
        end = millisec(end)
        if lastend > end:  # segment engulfed by a previous segment
            groups.append(g)
            g = []
        else:
            lastend = end
    if g:
        groups.append(g)
    # print(*groups, sep='\n')
    # Save the audio part corresponding to each diarization group.
    audio = AudioSegment.from_wav(prepended_input_file)
    gidx = -1
    os.makedirs("./data/split_files/", exist_ok=True)
    for g in groups:
        start = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[-1])[1]
        start = millisec(start)  # - spacermilli
        end = millisec(end)  # - spacermilli
        gidx += 1
        audio[start:end].export(
            "./data/split_files/" + str(gidx) + ".wav", format="wav"
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
    parser.add_argument(
        "--dont_diarize",
        action="store_true",
        help="Option not to diarize if this has been done already. Default is False, i.e., *do* diarize!",
    )
    parser.add_argument(
        "--dont_split_audio",
        action="store_true",
        help="Whether or not to split audio. Requires that a diarization file exist! Default is false--i.e., *do* split audio!",
    )
    parser.add_argument(
        "--num_speakers",
        type=int,
        default=None,
        required=False,
        help="Number of speakers in audio file. Default is None for unknown",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argv()

    os.makedirs("./results/diarization", exist_ok=True)
    diarization_file = f"./results/diarization/{args.qualitative_name}_diarization.txt"

    if not args.dont_diarize:
        print("Prepending silence to input audio")
        spacermilli = 2000
        spacer = AudioSegment.silent(duration=spacermilli)
        audio = AudioSegment.from_wav(args.input_audio_file)
        audio = spacer.append(audio, crossfade=0)
        prepared_audio_filepath = args.input_audio_file.replace(".wav", "_prep.wav")
        audio.export(prepared_audio_filepath, format="wav")

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.0",
            use_auth_token=hf_token,
        )

        # send pipeline to GPU (when available)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)

        # apply pretrained pipeline
        print("Now applying pretrained pipeline to input audio")
        diarization = pipeline(prepared_audio_filepath, num_speakers=args.num_speakers)

        print("Writing diarization file")
        with open(diarization_file, "w") as text_file:
            text_file.write(str(diarization))
        # # print the result
        # for turn, _, speaker in diarization.itertracks(yield_label=True):
        #     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    if not args.dont_split_audio:
        print("Splitting audio based on diarization file")
        group_and_save(
            diarization_file=diarization_file,
            prepended_input_file=prepared_audio_filepath,
        )
