# Side 1
echo "Now transcribing side 1..."
python3 run_faster_whisper.py \
    --input_audio_file ./data/original_files/david_and_ada_side_1.wav \
    --qualitative_name david_and_ada_side_1 ;

echo "Now diarizing side 1..."
python3 pyannote_diarize.py \
    --num_speakers 3 \
    --input_audio_file ./data/original_files/david_and_ada_side_1.wav \
    --qualitative_name david_and_ada_side_1 \
    --dont_split_audio ;

echo "Now stitching it all together for side 1..."
python3 stitch_diarization_and_transcription.py \
    --diarization_file ./results/diarization/david_and_ada_side_1_diarization.txt \
    --transcription_file ./results/non_diarized_transcriptions/david_and_ada_side_1_transcriptions.csv \
    --qualitative_name david_and_ada_side_1 ;

# Side 2
echo "Now transcribing side 2..."
python3 run_faster_whisper.py \
    --input_audio_file ./data/original_files/david_and_ada_side_2.wav \
    --qualitative_name david_and_ada_side_2 ;

echo "Now diarizing side 1..."
python3 pyannote_diarize.py \
    --num_speakers 3 \
    --input_audio_file ./data/original_files/david_and_ada_side_2.wav \
    --qualitative_name david_and_ada_side_2 \
    --dont_split_audio ;

echo "Now stitching it all together for side 2..."
python3 stitch_diarization_and_transcription.py \
    --diarization_file ./results/diarization/david_and_ada_side_2_diarization.txt \
    --transcription_file ./results/non_diarized_transcriptions/david_and_ada_side_2_transcriptions.csv \
    --qualitative_name david_and_ada_side_2 ;

echo "Done processing your audio!"