import requests
import librosa
from pydub import AudioSegment
from moviepy.editor import (
    AudioFileClip,
    VideoFileClip,
    concatenate_videoclips,
    CompositeVideoClip,
    vfx,
)
import subprocess
from pathlib import Path
import soundfile as sf
import os
import cv2
from urllib.parse import urlparse
from Wav2Lip.inference import main

# import ffmpeg
base_dir = Path(__file__).resolve().parent
from vits_audio import app
from minio import Minio
from dotenv import load_dotenv

load_dotenv()

# Retrieve MinIO credentials and configurations from environment variables
minio_server_url = urlparse(os.getenv("MINIO_EXTERNAL_BASE_URL")).netloc
minio_access_key = os.getenv("MINIO_ACCESS_KEY")
minio_secret_key = os.getenv("MINIO_SECRET_KEY")
voice_cloning_api = os.getenv("VOICE_CLONING_API")
bucket_name = "cm-user-videos"


# Initialize MinIO client
minio_client = Minio(
    minio_server_url,  # The MinIO server URL (replace with actual MinIO server IP or hostname)
    access_key=minio_access_key,  # MinIO access key
    secret_key=minio_secret_key,  # MinIO secret key
    secure=True,  # Use `True` for HTTPS; `False` for HTTP
)


def get_video_from_minio(name):
    return minio_client.get_object(bucket_name, f"user-lipsync-video/{name}.mp4")


def check_object_exists(object_name):
    try:
        # Try to get the object metadata
        minio_client.stat_object(bucket_name, object_name)
        return True  # Object exists
    except Exception as e:
        print(f"{e}")
        return False  # Object does not exist


# Check if bucket exists, otherwise create it
if not minio_client.bucket_exists(bucket_name):
    minio_client.make_bucket(bucket_name)
    print(f"Bucket '{bucket_name}' created")
else:
    print(f"Bucket '{bucket_name}' already exists")


# Upload a file to the bucket
def upload_file(local_file_path, object_name, content_type="video/mp4"):
    try:
        minio_client.fput_object(
            bucket_name, object_name, local_file_path, content_type
        )
        print(f"File '{local_file_path}' uploaded successfully as '{object_name}'")
    except Exception as err:
        print("Error occurred while uploading file:", err)


# Constants
ELEVEN_API_KEY = "sk_259d78a9db90a19c034f22f34d2f38775676631f664dd505"

CHUNK_SIZE = 1024
voice_id = "i22asromhds68FUNLHed"  # AAP ki adalat voice


def extract_audio(video_file, output_audio):
    # Check if the video file exists
    if os.path.exists(video_file):
        # Load the video and extract audio
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(output_audio)
        video.close()
        print(f"Audio extracted from {video_file} to {output_audio}")
        return output_audio
    else:
        print(f"{video_file} does not exist.")
        return None


def combine_audios(audio_files, output_combined_audio):
    combined_audio = AudioSegment.silent(
        duration=0
    )  # Start with an empty audio segment
    for audio_file in audio_files:
        if audio_file and os.path.exists(audio_file):
            audio_segment = AudioSegment.from_file(audio_file)
            combined_audio += audio_segment  # Append the audio to the combined audio
        else:
            print(f"{audio_file} does not exist or is empty.")

    # Export the combined audio
    combined_audio.export(output_combined_audio, format="mp3")
    print(f"Combined audio saved to {output_combined_audio}")


# Function to call ElevenLabs API
def voice_output(
    file_name,
    word,
    tts_suffix,
    temperature,
    length_penalty,
    repetition_penalty,
    top_k,
    top_p,
    sentence_split,
    coqui_tts_api
):
    # url = voice_cloning_api+coqui_tts_api
    # headers = {"accept": "application/json"}
    # params = {
    #     "tts_text": word,
    #     "tts_suffix": tts_suffix,
    #     "temperature": temperature,
    #     "length_penalty": length_penalty,
    #     "repetition_penalty": repetition_penalty,
    #     "top_k": top_k,
    #     "top_p": top_p,
    #     "sentence_split": sentence_split,
    # }

    # Make the POST request with streaming enabled

    
    audio_buffer = app.vits_hindi(txt=word)
    
    # Read the entire buffer at once
    audio_data = audio_buffer.read()

    # Write the audio data to a file in one go
    with open(f"{file_name}", "wb") as audio_file:
        audio_file.write(audio_data)
    # Simulate sending in chunks
    # chunk_size = 8192  # Size of each chunk in bytes
    # while True:
    #     chunk = audio_buffer.read(chunk_size)
    #     if not chunk:  # End of buffer
    #         break
    #     # Process the chunk (e.g., write to a file, send over a network)
    #     print(f"Sending chunk of size {len(chunk)} bytes")
    #     with open(f"{file_name}.wav", "ab") as audio_file:
    #         audio_file.write(chunk)

    # response = requests.post(url, headers=headers, params=params, stream=True)

    # # Check if the request was successful
    # if response.status_code == 200:
    #     # Open a file to save the audio content
    #     with open(file_name, "wb") as audio_file:
    #         for chunk in response.iter_content(chunk_size=8192):
    #             if chunk:  # Filter out keep-alive chunks
    #                 audio_file.write(chunk)
    #     print(f"Audio downloaded successfully as {file_name}")
    #     upload_file(file_name, f"/user-lipsync-video/lipsync_{word}.wav")
    #     print(
    #         "Uploaded {'final_result.mp4'} to S3 bucket {bucket_name}/user-lipsync-video/lipsync_{name_en}.wav"
    #     )
    # else:
    #     print(f"Failed to download audio. Status code: {response.status_code}")
    # url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    # headers = {"Accept": "audio/mpeg", "Content-Type": "application/json", "xi-api-key": ELEVEN_API_KEY}
    # data = {"text": word, "model_id": "eleven_multilingual_v2", "voice_settings": {"stability": 1, "similarity_boost": 1}}

    # response = requests.post(url, json=data, headers=headers)
    # with open(f"{file_name}", "wb") as f:
    #     for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
    #         if chunk:
    #             f.write(chunk)


def normalize_volume(generated_audio_file, output_file, output_length):
    # print("output_file:", output_file)
    original_audio_file = "static_audio.mp3"
    extract_audio("static_part_1.mp4", "static_part_1.mp3")
    extract_audio("static_part_3.mp4", "static_part_3.mp3")
    combine_audios(["static_part_1.mp3", "static_part_3.mp3"], original_audio_file)

    # Load original and generated audio
    original_audio = AudioSegment.from_file(original_audio_file)
    generated_audio = AudioSegment.from_file(generated_audio_file)

    # Calculate the loudness difference
    original_loudness = original_audio.dBFS
    generated_loudness = generated_audio.dBFS
    volume_difference = original_loudness - generated_loudness

    # Adjust the generated audio volume
    adjusted_generated_audio = generated_audio.apply_gain(volume_difference)

    # Check if the adjusted audio is longer or shorter than the desired output_length
    if len(adjusted_generated_audio) > output_length:
        # Clip the audio to the desired length
        adjusted_generated_audio = adjusted_generated_audio
    elif len(adjusted_generated_audio) < output_length:
        # Add silence to the end to match the output_length
        silence_to_add = AudioSegment.silent(
            duration=output_length - len(adjusted_generated_audio)
        )
        adjusted_generated_audio = adjusted_generated_audio + silence_to_add

    # Export the adjusted audio to the output file
    adjusted_generated_audio.export(output_file, format="wav")


# Define paths to input video and audio files
def lipsync_video(input_audio, input_video, name_en, model):
    # input_video - mp4, input_audio- wav
    print("input_video", input_video, input_audio)
    input_video = Path(input_video)
    input_audio = Path(os.path.join(base_dir,input_audio)) #Path(input_audio)

    # Ensure that the paths to input files exist
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio}")

    # Run the Wav2Lip inference script with appropriate arguments
    main(model, str(input_video), input_audio)

    upload_file(
        os.path.join(base_dir, "Wav2Lip/results/result_voice.mp4"),
        f"/user-lipsync-video/{name_en}.mp4",
    )
    return os.path.join(base_dir, "Wav2Lip/results/result_voice.mp4")


# def adjust_audio_video(input_video_file, input_audio_file, output_file):
#     # Load the video
#     video = VideoFileClip(input_video_file)
#     video_duration = video.duration  # in seconds (should be 1 minute)

#     # Load the audio
#     audio = AudioSegment.from_wav(input_audio_file)
#     audio_duration = len(audio) / 1000  # Convert from ms to seconds

#     print(f"Video duration: {video_duration} seconds")
#     print(f"Audio duration: {audio_duration} seconds")

#     # Check if audio is longer than the video
#     if audio_duration != video_duration:
#         # Stretch or contract the audio to match the video duration
#         speed_factor = audio_duration / video_duration
#         if speed_factor > 1:
#             print(f"Contracting the audio by factor: {speed_factor}")
#         else:
#             print(f"Stretching the audio by factor: {1/speed_factor}")

#         adjusted_audio = audio.speedup(playback_speed=speed_factor)

#         # Save the adjusted audio to a temporary file
#         adjusted_audio.export("temp_adjusted_audio.wav", format="wav")

#         # Load the adjusted audio and set it to the video
#         adjusted_audio_clip = AudioFileClip("temp_adjusted_audio.wav")
#         video = video.set_audio(adjusted_audio_clip)

#     # Write the output video with the adjusted audio
#     video.write_videofile(output_file, codec="libx264", audio_codec="aac")


# def adjust_audio_video(input_video_file, input_audio_file, output_video_file, output_audio_file):

#     video = VideoFileClip(input_video_file)
#     video_duration = video.duration

#     # Load the audio file
#     y, sr = librosa.load(input_audio_file, sr=None)
#     audio_duration = librosa.get_duration(y=y, sr=sr)

#     # Calculate the speed change factor
#     speed_factor = audio_duration / video_duration

#     # Time-stretch the audio without changing pitch
#     y_stretched = librosa.effects.time_stretch(y, rate=speed_factor)

#     # Save the stretched audio temporarily
#     sf.write(output_audio_file, y_stretched, sr)

#     # Load the stretched audio as an AudioFileClip
#     new_audioclip = AudioFileClip(output_audio_file)

#     # Set the audio of the video to the new stretched audio
#     video = video.set_audio(new_audioclip)

#     # Write the result to a file
#     video.write_videofile(output_video_file, codec='libx264', audio_codec='aac')


#     # Close the clips
#     video.close()
#     new_audioclip.close()
def adjust_audio_video(
    input_video_file, input_audio_file, output_video_file, output_audio_file
):
    video = VideoFileClip(input_video_file)
    video_duration = video.duration

    # Load the audio file
    y, sr = librosa.load(input_audio_file, sr=None)
    audio_duration = librosa.get_duration(y=y, sr=sr)

    # Time-stretch the audio without changing pitch
    y_stretched = librosa.effects.time_stretch(y, rate=1)

    # Save the stretched audio temporarily
    sf.write(output_audio_file, y_stretched, sr)
    # Load the video file and get its duration
    video = VideoFileClip(input_video_file)
    video_duration = video.duration

    # Load the audio file and get its duration
    y, sr = librosa.load(input_audio_file, sr=None)
    audio_duration = librosa.get_duration(y=y, sr=sr)

    # Calculate the speed change factor for the video
    speed_factor = video_duration / audio_duration

    # Adjust the video's playback speed to match the audio duration
    new_video = video.fx(vfx.speedx, factor=speed_factor)

    # Set the audio of the new video to the original audio file
    new_audio = AudioFileClip(input_audio_file)
    new_video = new_video.set_audio(new_audio)

    # Write the result to the output file
    new_video.write_videofile(output_video_file, codec="libx264", audio_codec="aac")

    # Close the clips
    video.close()
    new_video.close()
    new_audio.close()


# def cut_video(input_video_path, start_time, end_time, output_part1, output_part2, output_part3):
#     # Load the video
#     video = VideoFileClip(input_video_path)
#     start_time = start_time/1000
#     end_time = end_time/1000
#     # Part 1: from 0 to start_time
#     if 0!=start_time:
#         part1 = video.subclip(0, start_time)
#         part1.write_videofile(output_part1)
#         s3.upload_file(output_part1, S3_BUCKET, output_part1)

#     if start_time!=end_time:
#         # Part 2: from start_time to end_time
#         part2 = video.subclip(start_time, end_time)
#         part2.write_videofile(output_part2)
#         s3.upload_file(output_part2, S3_BUCKET, output_part2)

#     # Part 3: from end_time to the rest of the video
#     part3 = video.subclip(end_time, video.duration)
#     part3.write_videofile(output_part3)
#     s3.upload_file(output_part3, S3_BUCKET, output_part3)

#     # Close video resources
#     video.close()


def cut_video(
    input_video_path, start_time, end_time, output_part1, output_part2, output_part3
):
    video = VideoFileClip(input_video_path)
    start_time = start_time / 1000
    end_time = end_time / 1000
    print("cut video timings ", start_time, end_time, video.duration)
    # Part 1: from 0 to start_time

    part1 = video.subclip(0, start_time)

    # Part 2: from start_time to end_time
    part2 = video.subclip(start_time, end_time).without_audio()

    # Part 3: from end_time to the rest of the video
    part3 = video.subclip(end_time, video.duration)

    # Close video resources
    if start_time != 0:
        part1.write_videofile(output_part1, codec="libx264")
    part2.write_videofile(output_part2, codec="libx264")
    part3.write_videofile(output_part3, codec="libx264")
    video.close()
    part1.close()
    part2.close()
    part3.close()


def get_file_size(filepath):
    """Get the file size in MB"""
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert from bytes to MB
    return file_size


def compress_video(input_video_path, output_video_path, compression_ratio=1):
    """
    Compress the video using ffmpeg by adjusting bitrate and resolution.
    Tries to reduce the video size to be under or close to the target size.
    """
    # Load the video using moviepy to get its original details
    video = VideoFileClip(input_video_path)
    original_duration = video.duration

    # Calculate the target bitrate based on desired file size and video duration
    target_bitrate = (
        compression_ratio * 8 * 1024 * 1024
    ) / original_duration  # in bits per second

    # if os.path.exists(os.path.join(base_dir, output_video_path)):
    #     return
    # Use ffmpeg to compress the video

    command = [
        "ffmpeg",
        "-i",
        input_video_path,  # Input video file
        "-b:v",
        f"{int(target_bitrate)}",  # Set target bitrate
        "-b:a",
        "128k",  # Set audio bitrate
        "-vf",
        "scale=-2:480",  # Resize to 480p
        "-preset",
        "fast",  # Compression preset (can try 'slow' or 'veryslow' for better compression)
        "-crf",
        "28",  # Constant Rate Factor (controls quality and size)
        "-y",  # Overwrite output file without asking
        output_video_path,  # Output file
    ]

    # Run the FFmpeg command using subprocess
    try:
        subprocess.run(command, check=True)
        print(f"Video compressed and saved to {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while compressing the video: {e}")

    # Get the compressed video size
    compressed_size = get_file_size(output_video_path)
    print(f"Compressed video size: {compressed_size:.2f} MB")


# def video_stitching():
#     output_file = "final_result.mp4"
#     video_files = []
#     if os.path.exists(os.path.join(base_dir, "static_part_1.mp4")):
#         video_files.append(os.path.join(base_dir, "static_part_1.mp4"))
#         print("found", os.path.join(base_dir, "static_part_1.mp4"))
#         time.sleep(0.01)
#     else:
#         print("not found", os.path.join(base_dir, "static_part_1.mp4"))
#     video_files.append(os.path.join(base_dir, "Wav2Lip/results/result_voice.mp4"))
#     if os.path.exists(os.path.join(base_dir, "static_part_3.mp4")):
#         video_files.append(os.path.join(base_dir, "static_part_3.mp4"))
#         print("found", os.path.join(base_dir, "static_part_3.mp4"))
#         time.sleep(0.01)
#     else:
#         print("not found", os.path.join(base_dir, "static_part_3.mp4"))
#     clips = [VideoFileClip(video) for video in video_files]
#     final_clip = concatenate_videoclips(clips)
#     time.sleep(0.01)
#     final_clip.write_videofile(output_file, codec="libx264")
#     # Upload to S3
#     try:
#         print("video_files concatenate_videoclips", video_files)
#         s3.upload_file(output_file, S3_BUCKET, output_file)
#         print(f"Uploaded {output_file} to S3 bucket {S3_BUCKET}")
#     except Exception as e:
#         print(f"Failed to upload {output_file} to S3: {e}")


# def reencode_video(input_file, output_file):
#     """Re-encodes the video to ensure compatible audio format."""
#     print(f"********************Re-encodes   {input_file, output_file}      ******************")
#     try:
#         subprocess.run([
#             'ffmpeg',
#             '-i', input_file,
#             '-c:v', 'copy',
#             '-c:a', 'aac',
#             '-b:a', '128k',
#             output_file,
#             '-y'
#         ], check=True)
#         print(f"Re-encoded {input_file} to {output_file}.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error re-encoding {input_file}: {e}")

# def video_stitching():
#     """
#     Combines multiple MP4 videos into one using ffmpeg.

#     :param video_list: List of video file paths to combine.
#     :param output_file: The path to save the combined output video.
#     """
#     video_files = []
#     # if os.path.exists(os.path.join(base_dir, "static_part_1.mp4")):
#     #     video_files.append(os.path.join(base_dir, "static_part_1.mp4"))
#     #     print("found", os.path.join(base_dir, "static_part_1.mp4"))
#     #     time.sleep(0.01)
#     # else:
#     #     print("not found", os.path.join(base_dir, "static_part_1.mp4"))
#     video_files.append("file "+ os.path.join(base_dir, "Wav2Lip/results/rresult_voice.mp4"))
#     reencode_video(os.path.join(base_dir, "Wav2Lip/results/result_voice.mp4"), os.path.join(base_dir, "Wav2Lip/results/rresult_voice.mp4"))
#     if os.path.exists(os.path.join(base_dir, "static_part_3.mp4")):
#         video_files.append("file "+ os.path.join(base_dir, "rstatic_part_3.mp4"))
#         reencode_video(os.path.join(base_dir, "static_part_3.mp4"), os.path.join(base_dir, "rstatic_part_3.mp4"))
#         print("found", os.path.join(base_dir, "static_part_3.mp4"))
#     else:
#         print("not found", os.path.join(base_dir, "static_part_3.mp4"))

#     try:
#         # Create a text file listing the videos to concatenate
#         with open('file_list.txt', 'w') as f:
#             video_list_content = "\n".join(video_files)
#             f.write(video_list_content)

#         # Run the ffmpeg command to concatenate the videos
#         subprocess.run([
#             'ffmpeg',
#             '-f', 'concat',
#             '-safe', '0',
#             '-i', 'file_list.txt',
#             '-c:v', 'libx264',  # Re-encode video using libx264
#             '-c:a', 'aac',  # Re-encode audio using AAC
#             '-b:a', '128k',  # Set audio bitrate to 128 kbps
#             # '-c', 'copy',
#             'final_result.mp4',
#             '-y'
#         ], check=True)

#         print(f"Videos combined successfully into final_result.mp4")


#     except subprocess.CalledProcessError as e:
#         print(f"Error combining videos: {e}")
def get_video_duration(video_path):
    """
    Returns the duration of a video in seconds.

    :param video_path: str, path to the video file
    :return: float, duration of the video in seconds
    """
    try:
        if not os.path.exists(video_path):
            return 0
        with VideoFileClip(video_path) as video:
            duration = video.duration
            print(f"video_path duration {duration} s")
        return duration
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return 0


def reencode_video(input_file, output_file):
    """Re-encodes the video to ensure compatible audio format."""
    print(
        f"********************Re-encodes   {input_file, output_file}      ******************"
    )
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i",
                input_file,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "128k",
                output_file,
                "-y",
            ],
            check=True,
        )
        print(f"Re-encoded {input_file} to {output_file}.")
    except subprocess.CalledProcessError as e:
        print(f"Error re-encoding {input_file}: {e}")


def process_videos(original_video_path, second_video_path, output_video_path):
    # Open both videos
    cap1 = cv2.VideoCapture(original_video_path)
    cap2 = cv2.VideoCapture(second_video_path)

    # Get video properties (from both videos)
    total_frames_original = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total_frames_second = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original Video Frames: {total_frames_original}")
    print(f"Second Video Frames: {total_frames_second}")

    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    width = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare video writer for adjusted second video
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Frame reduction logic for second video
    if total_frames_second > total_frames_original:
        # Calculate the number of frames to remove
        frames_to_remove = total_frames_second - total_frames_original
        remove_interval = total_frames_second // frames_to_remove

        current_frame = 0
        frame_index = 0
        while cap2.isOpened():
            ret2, frame2 = cap2.read()
            if not ret2:
                break

            # Remove frames at intervals, keeping the last frame intact
            if (
                frame_index < total_frames_second - 1
                and current_frame % remove_interval == 0
            ):
                frame_index += 1
                current_frame += 1
                continue  # Skip this frame

            out.write(frame2)  # Write frame to output video
            frame_index += 1
            current_frame += 1

    else:
        # Simply copy frames if no reduction is needed
        while cap2.isOpened():
            ret2, frame2 = cap2.read()
            if not ret2:
                break
            out.write(frame2)

    # Read the last frame of both videos and save as image
    cap1.set(cv2.CAP_PROP_POS_FRAMES, total_frames_original - 1)
    ret1, last_frame1 = cap1.read()
    if ret1:
        cv2.imwrite("last_frame_original.jpg", last_frame1)

    cap2.set(cv2.CAP_PROP_POS_FRAMES, total_frames_second - 1)
    ret2, last_frame2 = cap2.read()
    if ret2:
        cv2.imwrite("last_frame_second_video.jpg", last_frame2)

    # Release resources
    cap1.release()
    cap2.release()
    out.release()


def video_stitching(cropped_from, crop_supported, name, name_en):
    """
    Combines multiple MP4 videos into one using ffmpeg.

    :param video_list: List of video file paths to combine.
    :param output_file: The path to save the combined output video.
    """

    video_files = []
    original_video_path = os.path.join(base_dir, "cropped_video.mp4")
    model_result_path = os.path.join(base_dir, "Wav2Lip/results/result_voice.mp4")

    process_videos(original_video_path, model_result_path, "fps_corrected_wav2lip.mp4")
    part1 = os.path.join(base_dir, "static_part_1.mp4")
    part2 = os.path.join(base_dir, "fps_corrected_wav2lip.mp4")
    part3 = os.path.join(base_dir, "static_part_3.mp4")
    re_part1 = os.path.join(base_dir, "rstatic_part_1.mp4")
    re_part2 = os.path.join(base_dir, "Wav2Lip/results/rresult_voice.mp4")
    re_part3 = os.path.join(base_dir, "rstatic_part_3.mp4")
    if crop_supported:
        overlay_video(cropped_from, part2, "overlayed_video.mp4")
        part2 = os.path.join(base_dir, "overlayed_video.mp4")
        re_part2 = os.path.join(base_dir, "roverlayed_video.mp4")

    print(
        f"get_video_duration(part1) {get_video_duration(part1)} get_video_duration(part2) {get_video_duration(part2)} get_video_duration(part3) {get_video_duration(part3)}"
    )

    if get_video_duration(part1):
        reencode_video(part1, re_part1)
        video_files.append("file " + re_part1)
        print(f"Added {re_part1} to stitching")

    if get_video_duration(part2):
        reencode_video(part2, re_part2)
        video_files.append("file " + re_part2)
        print(f"Added {re_part2} to stitching")

    if get_video_duration(part3):
        reencode_video(part3, re_part3)
        video_files.append("file " + re_part3)
        print(f"Added {re_part3} to stitching")

    try:
        # Create a text file listing the videos to concatenate
        with open("file_list.txt", "w") as f:
            video_list_content = "\n".join(video_files)
            f.write(video_list_content)

        # Run the ffmpeg command to concatenate the videos
        subprocess.run(
            [
                "ffmpeg",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                "file_list.txt",
                "-c:v",
                "libx264",  # Re-encode video using libx264
                "-c:a",
                "aac",  # Re-encode audio using AAC
                "-b:a",
                "128k",  # Set audio bitrate to 128 kbps
                # '-c', 'copy',
                "final_result.mp4",
                "-y",
            ],
            check=True,
        )

        print("Videos combined successfully into final_result.mp4")

    except subprocess.CalledProcessError as e:
        print(f"Error combining videos: {e}")
    try:
        upload_file("final_result.mp4", f"/user-lipsync-video/{name_en}.mp4")
        print(
            "Uploaded {'final_result.mp4'} to S3 bucket {bucket_name}/user-lipsync-video/{name_en}.mp4"
        )
    except Exception as e:
        print("Failed to upload 'final_result.mp4' to S3: {e}")


def crop_video(input_video):
    # Open the video file
    video_path = input_video  # Replace with the path to your video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
    else:
        # Get the video details
        fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Video width
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Video height
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames

        print(f"Video Resolution: {width}x{height}")
        print(f"FPS: {fps}, Total Frames: {total_frames}")

        # Define the crop area (x1, y1, x2, y2)
        x1, y1 = width // 3, 0  # top-left corner (x, y)
        x2, y2 = 2 * width // 3, 2 * height // 3  # bottom-right corner (x, y)

        # Define the codec and create VideoWriter object to save the cropped video
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for saving as MP4
        out = cv2.VideoWriter("cropped_video.mp4", fourcc, fps, (x2 - x1, y2 - y1))

        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # If no more frames, exit loop

            # Crop the frame (slice the frame array)
            cropped_frame = frame[y1:y2, x1:x2]

            # Display the cropped frame (optional)
            # cv2.imshow('Cropped Frame', cropped_frame)

            # Write the cropped frame to the output video file
            out.write(cropped_frame)

            # Press 'q' to stop the video playback (optional)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Release the video capture and writer objects
        cap.release()
        out.release()

        # Close any open OpenCV windows
        cv2.destroyAllWindows()

        print("Video cropping and saving completed!")


def overlay_video(original_video_path, cropped_video_path, output_video_path):
    print(
        f"original_video_path {original_video_path}, cropped_video_path {cropped_video_path}, output_video_path {output_video_path}"
    )
    # Open the original and cropped videos
    # Load the original video
    original = VideoFileClip(original_video_path)

    # Load the cropped video
    cropped = VideoFileClip(cropped_video_path)

    # Get dimensions of the original video
    width, height = original.size

    # Define the position for the overlay
    x1, y1 = width // 3, 0  # top-left corner (x, y)
    x2, y2 = 2 * width // 3, 2 * height // 3

    # Resize the cropped video to match the overlay area
    cropped = cropped.resize((x2 - x1, y2 - y1))

    # Create a composite video with the original video and the overlay
    final_video = CompositeVideoClip([original, cropped.set_position((x1, y1))])

    # Write the final video to a file, ensuring audio is included
    final_video.write_videofile(output_video_path, codec="libx264", audio_codec="aac")
