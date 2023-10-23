import argparse
import pandas as pd
import re
import os

SPEAKER_MAPPING = {
    "SPEAKER_00": "RACHEL",
    "SPEAKER_01": "ADA",
    "SPEAKER_02": "DAVID",
    "SPEAKER_03": "DAVID",
    "SPEAKER_04": "ADA",
}


def get_diarization_df(diarization_file: str):
    with open(diarization_file, "r") as f:
        diarization = f.read().split("\n")
    d = {
        "speaker": [],
        "start": [],
        "end": [],
    }
    for line in diarization:
        timestamps = re.findall("\d\d:\d\d:\d\d\.\d*", line)
        start = (
            pd.to_datetime(timestamps[0]) - pd.to_datetime("00:00:00")
        ).total_seconds()
        end = (
            pd.to_datetime(timestamps[1]) - pd.to_datetime("00:00:00")
        ).total_seconds()
        speaker = re.findall("SPEAKER_\d+", line)[0]
        d["start"].append(start)
        d["end"].append(end)
        d["speaker"].append(speaker)
    return pd.DataFrame(d)


def get_dialogue(
    transcription: pd.core.frame.DataFrame,
    diarization: pd.core.frame.DataFrame,
):
    speakers = []
    for i in range(len(transcription)):
        word_start = transcription.iloc[i]["word_start"]
        word_end = transcription.iloc[i]["word_end"]
        filtered = diarization[
            (word_start >= diarization["start"]) & (word_end < diarization["end"])
        ]
        if len(filtered) > 0:
            raw_speaker = filtered.iloc[0]["speaker"]
        else:
            raw_speaker = None
        speakers.append(SPEAKER_MAPPING.get(raw_speaker, "UNKNOWN"))

    transcription["speaker"] = speakers

    transcription["line"] = (
        transcription["speaker"] != transcription["speaker"].shift()
    ).cumsum()

    transcription = (
        transcription.groupby("line")
        .agg(
            {
                "speaker": "first",
                "word": lambda x: " ".join(x),
                # "word_start": "min",
                # "word_end": "max",
            }
        )
        .reset_index()
    )

    output = ""
    for i in range(len(transcription)):
        speaker = transcription.iloc[i]["speaker"]
        # line_start = transcription.iloc[i]["word_start"]
        # line_end = transcription.iloc[i]["word_end"]
        text = transcription.iloc[i]["word"]
        # output += f"\n{speaker} ({line_start} - {line_end}):\n{text}\n"
        output += f"\n{speaker}:\n{text}\n"

    return output


def parse_argv() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diarization_file",
        required=False,
        default="./results/diarization/david_and_ada_side_1_short_diarization.txt",
        type=str,
        help="Path to txt file containing diarization",
    )
    parser.add_argument(
        "--transcription_file",
        required=False,
        default="./results/non_diarized_transcriptions/david_and_ada_side_1_short_transcriptions.csv",
        type=str,
        help="Path to csv mapping each word to its start and end timestamps",
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

    transcription = pd.read_csv(args.transcription_file)

    diarization = get_diarization_df(args.diarization_file)

    dialogue = get_dialogue(
        transcription=transcription,
        diarization=diarization,
    )

    os.makedirs("./results/dialogues", exist_ok=True)
    output_file = f"./results/dialogues/{args.qualitative_name}_dialogue.txt"
    if f"{args.qualitative_name}_dialogue.txt" in os.listdir("./results/dialogues"):
        os.remove(output_file)
    with open(output_file, "w") as f:
        f.write(dialogue)
