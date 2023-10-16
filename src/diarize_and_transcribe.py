import argparse
from pyannote.audio import Pipeline
from pydub import AudioSegment
import whisper
import torch
import re
import os
import json
from tqdm import tqdm
from mysecrets import hf_token

speakers = {
    "SPEAKER_00": "Ada",
    "SPEAKER_01": "David",
    "SPEAKER_02": "Rachel",
}


def timeStr(t):
    return "{0:02d}:{1:02d}:{2:06.2f}".format(
        round(t // 3600), round(t % 3600 // 60), t % 60
    )


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
    os.makedirs("./data/diarization/split_files/audio/", exist_ok=True)
    for g in groups:
        start = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        end = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[-1])[1]
        start = millisec(start)  # - spacermilli
        end = millisec(end)  # - spacermilli
        gidx += 1
        audio[start:end].export(
            "./data/diarization/split_files/audio/" + str(gidx) + ".wav", format="wav"
        )
    return groups


def transcribe_split_groups(groups, model) -> None:
    os.makedirs("./data/diarization/split_files/transcriptions/", exist_ok=True)
    for i in tqdm(range(len(groups))):
        print(f"transcribing group {i}")
        audiof = "./data/diarization/split_files/audio/" + str(i) + ".wav"
        result = model.transcribe(audio=audiof, language="en", word_timestamps=True)
        with open(
            "./data/diarization/split_files/transcriptions/" + str(i) + ".json", "w"
        ) as outfile:
            json.dump(
                result,
                outfile,
                indent=4,
            )
    return


def transcibe_all(groups, speakers):
    txt = list("")
    gidx = -1
    for g in groups:
        shift = re.findall("[0-9]+:[0-9]+:[0-9]+\.[0-9]+", string=g[0])[0]
        shift = millisec(shift) - spacermilli  # the start time in the original video
        shift = max(shift, 0)

        gidx += 1

        captions = json.load(
            open("./data/diarization/split_files/transcriptions/" + str(gidx) + ".json")
        )["segments"]

        if captions:
            speaker = g[0].split()[-1]
            if speaker in speakers:
                speaker = speakers[speaker]

            for c in captions:
                start = shift + c["start"] * 1000.0
                start = start / 1000.0  # time resolution ot youtube is Second.
                end = (shift + c["end"] * 1000.0) / 1000.0
                txt.append(
                    f'[{timeStr(start)} --> {timeStr(end)}] [{speaker}] {c["text"]}\n'
                )

    os.makedirs("./results/diarization/", exist_ok=True)
    with open(f"./results/diarization/captions.txt", "w", encoding="utf-8") as file:
        s = "".join(txt)
        file.write(s)
        print("captions saved to ./results/diarization/captions.txt")
        # print(s+'\n')

    return


def parse_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_audio_file",
        required=False,
        default="./data/original_files/david_and_ada_side_1_short.wav",
        type=str,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_argv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = whisper.load_model("large", device=device)

    print("Prepending silence to input audio")
    spacermilli = 2000
    spacer = AudioSegment.silent(duration=spacermilli)
    audio = AudioSegment.from_wav(args.input_audio_file)
    audio = spacer.append(audio, crossfade=0)
    input_audio_with_silence = args.input_audio_file.replace(".wav", "_prep.wav")
    audio.export(input_audio_with_silence, format="wav")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.0",
        use_auth_token=hf_token,
    )

    # send pipeline to GPU (when available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)

    # apply pretrained pipeline
    print("Now applying pretrained diarization pipeline to input audio")
    diarization = pipeline(input_audio_with_silence)

    print("Writing diarization file")
    with open("./data/diarization/diarization.txt", "w") as text_file:
        text_file.write(str(diarization))
    # # print the result
    # for turn, _, speaker in diarization.itertracks(yield_label=True):
    #     print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")

    print("Splitting audio based on diarization file")
    groups = group_and_save(
        diarization_file="./data/diarization/diarization.txt",
        prepended_input_file=input_audio_with_silence,
    )

    print("Transcribing split groups")
    transcribe_split_groups(groups=groups, model=model)
    print("Done transcribing split groups; now stitching them all together")
    transcibe_all(groups=groups, speakers=speakers)
