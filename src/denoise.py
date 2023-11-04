import argparse
import os

from scipy.io import wavfile
import noisereduce as nr


def denoise_and_write(input_file_path: str, out_path: str):
    rate, data = wavfile.read(input_file_path)
    reduced_noise = nr.reduce_noise(
        y=data,
        sr=rate,
        stationary=True,
        freq_mask_smooth_hz=200,
        n_std_thresh_stationary=0.3,
    )
    wavfile.write(out_path, rate, reduced_noise)


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

    os.makedirs("./results/denoising", exist_ok=True)
    denoise_and_write(
        input_file_path=args.input_audio_file,
        out_path=f"./results/denoising/{args.qualitative_name}_denoised.wav",
    )
