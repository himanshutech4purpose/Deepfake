
import requests
import librosa
from pydub import AudioSegment
from moviepy.editor import AudioFileClip, VideoFileClip, concatenate_videoclips, CompositeVideoClip, vfx
import subprocess
from pathlib import Path
import soundfile as sf
import os
import cv2
from urllib.parse import urlparse
from Wav2Lip.inference import main
# import ffmpeg
base_dir = Path(__file__).resolve().parent

from minio import Minio
from dotenv import load_dotenv
load_dotenv()

# Retrieve MinIO credentials and configurations from environment variables
minio_server_url = urlparse(os.getenv("MINIO_EXTERNAL_BASE_URL")).netloc
minio_access_key = os.getenv("MINIO_ACCESS_KEY")
minio_secret_key = os.getenv("MINIO_SECRET_KEY")
voice_cloning_api = os.getenv("VOICE_CLONING_API")
bucket_name = 'cm-user-videos'


# Initialize MinIO client
minio_client = Minio(
    minio_server_url,  # The MinIO server URL (replace with actual MinIO server IP or hostname)
    access_key=minio_access_key,  # MinIO access key
    secret_key=minio_secret_key,  # MinIO secret key
    secure=True  # Use `True` for HTTPS; `False` for HTTP
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
        minio_client.fput_object(bucket_name, object_name, local_file_path, content_type)
        print(f"File '{local_file_path}' uploaded successfully as '{object_name}'")
    except Exception as err:
        print("Error occurred while uploading file:", err)

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

# Function to call ElevenLabs API
def voice_output(file_name, word):

    url = voice_cloning_api
    headers = {"accept": "application/json"}
    params = {"tts_text": word}

    # Make the POST request with streaming enabled
    response = requests.post(url, headers=headers, params=params, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a file to save the audio content
        with open(file_name, "wb") as audio_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # Filter out keep-alive chunks
                    audio_file.write(chunk)
        print(f"Audio downloaded successfully as {file_name}")
        upload_file(file_name, f'/user-lipsync-video/lipsync_{word}.wav')
        print("Uploaded {'final_result.mp4'} to S3 bucket {bucket_name}/user-lipsync-video/lipsync_{name_en}.wav")
    else:
        print(f"Failed to download audio. Status code: {response.status_code}")


# Define paths to input video and audio files
def lipsync_video(input_audio, input_video, name_en, model):
    # input_video - mp4, input_audio- wav
    input_video = Path(input_video)
    input_audio = Path(os.path.join(base_dir,input_audio)) #Path(input_audio)
    print("input_video", input_video, input_audio)

    # Ensure that the paths to input files exist
    if not input_video.exists():
        raise FileNotFoundError(f"Input video not found: {input_video}")
    if not input_audio.exists():
        raise FileNotFoundError(f"Input audio not found: {input_audio}")
    main(model, str(input_video), input_audio)

    # Run the Wav2Lip inference script with appropriate arguments
    # subprocess.run(
    #     [
    #         "python3",
    #         "Wav2Lip/inference.py",
    #         "--checkpoint_path",
    #         "Wav2Lip/checkpoints/wav2lip_gan.pth",
    #         "--face",
    #         str(input_video),
    #         "--audio",
    #         str(input_audio),
    #         # "--wav2lip_batch_size",
    #         # "1",
    #         # "--face_det_batch_size",
    #         # "1",
    #     ],
    #     check=True,
    # )
    
    upload_file(os.path.join(base_dir, "Wav2Lip/results/result_voice.mp4"), f'/user-lipsync-video/{name_en}.mp4')
    return os.path.join(base_dir, "Wav2Lip/results/result_voice.mp4")





def get_file_size(filepath):
    """Get the file size in MB"""
    file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert from bytes to MB
    return file_size

def compress_video(input_video_path, output_video_path, target_size_mb=1):
    """
    Compress the video using ffmpeg by adjusting bitrate and resolution.
    Tries to reduce the video size to be under or close to the target size.
    """
    # Load the video using moviepy to get its original details
    video = VideoFileClip(input_video_path)
    original_duration = video.duration

    # Calculate the target bitrate based on desired file size and video duration
    target_bitrate = (target_size_mb * 8 * 1024 * 1024) / original_duration  # in bits per second

    # if os.path.exists(os.path.join(base_dir, output_video_path)):
    #     return
    # Use ffmpeg to compress the video

    command = [
        'ffmpeg',
        '-i', input_video_path,  # Input video file
        '-b:v', f'{int(target_bitrate)}',  # Set target bitrate
        '-b:a', '128k',  # Set audio bitrate
        '-vf', 'scale=-2:480',  # Resize to 480p
        '-preset', 'fast',  # Compression preset (can try 'slow' or 'veryslow' for better compression)
        '-crf', '28',  # Constant Rate Factor (controls quality and size)
        '-y',  # Overwrite output file without asking
        output_video_path  # Output file
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

