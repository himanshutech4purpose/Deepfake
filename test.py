from pathlib import Path
import os


from typing import Optional

from pydub import AudioSegment
from moviepy.editor import AudioFileClip, VideoFileClip
from preprocessing import voice_output, normalize_volume, lipsync_video, adjust_audio_video, cut_video, compress_video, video_stitching

# Constants
ELEVEN_API_KEY = "sk_259d78a9db90a19c034f22f34d2f38775676631f664dd505"
S3_BUCKET = "himanshupersonals3"
S3_ORIGINAL_AUDIO = "input_audio_part_2.mp3"  # The original audio file in your S3 bucket

CHUNK_SIZE = 1024
voice_id = "i22asromhds68FUNLHed"  # AAP ki adalat voice

base_dir = Path(__file__).resolve().parent



def get_video_length_ms(file_path):
    video = VideoFileClip(file_path)
    duration_in_seconds = video.duration
    duration_in_milliseconds = duration_in_seconds * 1000
    return duration_in_milliseconds



def main(name, start_time, end_time, audio_length_adjustment):
    # constant
    original = "original.mp4"
    input_video = "input_video.mp4" # mp4 video require for wav2lip model
    elevenlabs_audio = "elevenlabs_audio.mp3"
    elevenlabs_audio_adjusted_wav = "elevenlabs_audio_adjusted.wav"
    wav2lip_video_input = "wav2lip_video_input.mp4"
    wav2lip_audio_input = "wav2lip_audio_input.wav"
    print(name)

    # step-1 compress the video
    compress_video(original, input_video)
    print("*********************  step-1 compress the video Completed  ***********************")

    # step-2 cut the video in 3 parts
    cut_video(
    input_video_path=input_video,
    start_time=start_time,
    end_time=end_time,
    output_part1="static_part_1.mp4",
    output_part2=input_video,
    output_part3="static_part_3.mp4"
    )
    print("*********************  step-2 cut the video in 3 parts Completed  ***********************")

    # Step-3 Generate voice output using ElevenLabs API
    length_ms = get_video_length_ms(input_video)
    voice_output(file_name=elevenlabs_audio, word=name)
    print("*********************  # Step-3 Generate voice output using ElevenLabs API Completed  ***********************")

    # Step-4 Normalize volume
    normalize_volume(elevenlabs_audio, elevenlabs_audio_adjusted_wav, length_ms)
    print("*********************  # Step-4 Normalize volume Completed  ***********************")

    # Step-5 Adjust the audio video length
    if audio_length_adjustment:
        adjust_audio_video(input_video, elevenlabs_audio_adjusted_wav , wav2lip_video_input, wav2lip_audio_input)
        print("*********************  # step-4 Adjust the audio video length ***********************")

    # Step-5 LipSync using wav2lip
    if audio_length_adjustment:
        lipsync_video(wav2lip_audio_input, wav2lip_video_input)
    else:
        lipsync_video(elevenlabs_audio_adjusted_wav, input_video)

    # Step-6 Stitch the video back
    video_stitching()
    print("*********************  # Step-6 Stitch the video back ***********************")

# @app.post("/get_video")
# async def get_video(request: VideoRequest = None):
#     # Construct the file path
#     main(name=request.name, start_time=request.start_time, end_time=request.end_time, audio_length_adjustment=request.audio_length_adjustment)
#     return {"status_code": 200, "url":"https://himanshupersonals3.s3.amazonaws.com/final_result.mp4"}





if __name__ == "__main__":
    main(name="Raj", start_time=0, end_time=700, audio_length_adjustment=True)
