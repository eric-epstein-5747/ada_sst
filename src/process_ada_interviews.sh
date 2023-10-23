# Side 1
python3 run_faster_whisper.py --input_audio_file ./data/original_files/david_and_ada_side_1.wav --qualitative_name david_and_ada_side_1 ;
python3 pyannote_diarize.py --input_audio_file ./data/original_files/david_and_ada_side_1.wav --qualitative_name david_and_ada_side_1 --dont_split_audio ;
python3 stitch_diarization_and_transcription.py --diarization_file ./results/diarization/david_and_ada_side_1_diarization.txt --transcription_file ./results/non_diarized_transcriptions/david_and_ada_side_1_transcriptions.csv --qualitative_name david_and_ada_side_1 ;

# Side 2
python3 run_faster_whisper.py --input_audio_file ./data/original_files/david_and_ada_side_2.wav --qualitative_name david_and_ada_side_2 ;
python3 pyannote_diarize.py --input_audio_file ./data/original_files/david_and_ada_side_2.wav --qualitative_name david_and_ada_side_2 --dont_split_audio ;
python3 stitch_diarization_and_transcription.py --diarization_file ./results/diarization/david_and_ada_side_2_diarization.txt --transcription_file ./results/non_diarized_transcriptions/david_and_ada_side_2_transcriptions.csv --qualitative_name david_and_ada_side_2 ;

