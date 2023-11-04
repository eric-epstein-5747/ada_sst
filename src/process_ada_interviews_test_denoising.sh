# Side 1
echo "Now denoising..."
python3 denoise.py \
    --input_audio_file ./data/original_files/david_and_ada_side_1_short.wav \
    --qualitative_name david_and_ada_side_1_short ;

echo "Now transcribing..."
python3 run_faster_whisper.py \
    --input_audio_file ./results/denoised/david_and_ada_side_1_short.wav \
    --qualitative_name david_and_ada_side_1_short ;

echo "Now diarizing..."
python3 pyannote_diarize.py \
    --num_speakers 4 \
    --input_audio_file ./results/denoised/david_and_ada_side_1_short.wav \
    --qualitative_name david_and_ada_side_1_short \
    --dont_split_audio ;

echo "Now stitching it all together..."
python3 stitch_diarization_and_transcription.py \
    --diarization_file ./results/diarization/david_and_ada_side_1_short_diarization.txt \
    --transcription_file ./results/non_diarized_transcriptions/david_and_ada_side_1_short_transcriptions.csv \
    --qualitative_name david_and_ada_side_1_short ;

echo "Done processing your audio!"