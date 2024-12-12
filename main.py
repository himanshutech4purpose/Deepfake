from pathlib import Path
import os
import time
from Wav2Lip.inference import load_model
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional

from pydub import AudioSegment
from moviepy.editor import AudioFileClip, VideoFileClip
from preprocessing import voice_output, lipsync_video, compress_video, check_object_exists, get_video_from_minio
from fastapi.responses import  StreamingResponse

app = FastAPI()
base_dir = Path(__file__).resolve().parent

class VideoRequest(BaseModel):
    name: str # Name in hindi
    name_en: str  # Name in english
    start_time: Optional[int] = 0
    end_time:Optional[int] = 700
    audio_length_adjustment:bool = True
    crop_supported:bool = True


# Global variable to store the model
model = None


# Event to load the model at startup
@app.on_event("startup")
async def pre_load_model():
    global model
    try:
        # Load your model here
        model_path = "Wav2Lip/checkpoints/wav2lip_gan.pth"
        model = load_model(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Failed to load the model.")


def predict(name, name_en, start_time, end_time, audio_length_adjustment, crop_supported):
    global model

    original = "original.mp4"
    input_video_compressed = "input_video_compressed.mp4"
    elevenlabs_audio = "elevenlabs_audio.wav"


    # step-1 compress the video
    compress_video(original, input_video_compressed)
    print("*********************  step-1 compress the video Completed  ***********************")

    voice_output(file_name=elevenlabs_audio, word=name)

    # Record start time
    start_time = time.time()

    # Call the function
    final_video_path = lipsync_video(elevenlabs_audio, input_video_compressed, name_en, model)

    # Record end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Time taken to prediction: {elapsed_time:.4f} seconds")

    return final_video_path

@app.post("/api/synthesize/video")
async def get_video(request: VideoRequest = None):

    if check_object_exists(f"user-lipsync-video/{request.name_en}.mp4"):
        try:
            response = get_video_from_minio(request.name_en)
        except Exception as e:
            return {"error": str(e)}
        
        # Set up the StreamingResponse
        video_response = StreamingResponse(
            response.stream(32*1024),  # Stream in chunks of 32 KB
            media_type="video/mp4"
        )
        
        # video_response.headers["Content-Disposition"] = f"inline; filename={request.name}.mp4"
        return video_response
    else:
        final_video_path = predict(name=request.name, name_en=request.name_en, start_time=request.start_time, end_time=request.end_time, audio_length_adjustment=request.audio_length_adjustment, crop_supported=request.crop_supported)
        
        
        video_file = open(final_video_path, mode="rb")
        print("video file location", os.path.join(base_dir, "final_result.mp4"))
        video_response = StreamingResponse(video_file, media_type="video/mp4")


        return video_response


@app.post("/api/v1/video/synthesize")
async def synthesize_video(request: VideoRequest = None):
    if check_object_exists(f"user-lipsync-video/{request.name}.mp4"):
        pass
    else:
        predict(name=request.name, name_en=request.name_en, start_time=request.start_time, end_time=request.end_time, audio_length_adjustment=request.audio_length_adjustment, crop_supported=request.crop_supported)
    return {"status_code": 200, "status_message":"success", "url":f"https://cdn-api.dev.ks.samagra.io/cm-user-videos/user-lipsync-video/{request.name_en}.mp4"}


@app.get("/health")
def health_check():
    return {"status_code": 200}

