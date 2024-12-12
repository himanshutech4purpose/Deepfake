from pathlib import Path
import os
import time

from Wav2Lip.inference import load_model
from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional

from pydub import AudioSegment
from moviepy.editor import AudioFileClip, VideoFileClip
from preprocessing_old import (
    voice_output,
    normalize_volume,
    lipsync_video,
    adjust_audio_video,
    cut_video,
    compress_video,
    video_stitching,
    crop_video,
    check_object_exists,
    get_video_from_minio,
)
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Constants

CHUNK_SIZE = 1024
voice_id = "i22asromhds68FUNLHed"  # AAP ki adalat voice

app = FastAPI()
base_dir = Path(__file__).resolve().parent


class VideoRequest(BaseModel):
    name: Optional[str] = "शुभम"  # Name can be "xyz" or an optional string
    name_en: Optional[str] = "shubham"  # Name can be "xyz" or an optional string
    audio_length_adjustment: bool = True
    crop_supported: bool = True
    tts_suffix: Optional[str] = "            जी के लिए उत्तर प्रदेश "
    temperature: Optional[float] = 0.1
    length_penalty: Optional[int] = 5
    repetition_penalty: Optional[float] = 10.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.1
    sentence_split: Optional[int] = 0
    coqui_tts_api: Optional[str] = ""

class prepare_video_request(BaseModel):
    start_time: Optional[int] = 0
    end_time: Optional[int] = 700
    compression_ratio: Optional[int] = 1




def get_video_length_ms(file_path):
    video = VideoFileClip(file_path)
    duration_in_seconds = video.duration
    duration_in_milliseconds = duration_in_seconds * 1000
    return duration_in_milliseconds

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


def main(
    name,
    name_en,
    audio_length_adjustment,
    crop_supported,
    tts_suffix,
    temperature,
    length_penalty,
    repetition_penalty,
    top_k,
    top_p,
    sentence_split,
    coqui_tts_api
):
    # constant
    global model
    input_video = "input_video.mp4"  # mp4 video require for wav2lip model
    elevenlabs_audio = "elevenlabs_audio.wav"
    elevenlabs_audio_adjusted_wav = "elevenlabs_audio_adjusted.wav"
    wav2lip_video_input = "wav2lip_video_input1.mp4"
    wav2lip_audio_input = "wav2lip_audio_input.wav"
    cropped_video = "cropped_video.mp4"
    cropped_from = None



    # Step-3 Generate voice output using ElevenLabs API
    length_ms = get_video_length_ms(input_video)
    start_time = time.time()
    voice_output(
        file_name=elevenlabs_audio,
        word=name,
        tts_suffix=tts_suffix,
        temperature=temperature,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        sentence_split=sentence_split,
        coqui_tts_api=coqui_tts_api
    )
    end_time = time.time()
    print(f"Time taken in audio generaion from coqui-tts {end_time - start_time}")
    print(
        "*********************  # Step-3 Generate voice output using ElevenLabs API Completed  ***********************"
    )

    # Step-4 Normalize volume
    normalize_volume(elevenlabs_audio, elevenlabs_audio_adjusted_wav, length_ms)
    print(
        "*********************  # Step-4 Normalize volume Completed  ***********************"
    )

    # Step-5 Adjust the audio video length
    if audio_length_adjustment:
        adjust_audio_video(
            input_video,
            elevenlabs_audio_adjusted_wav,
            wav2lip_video_input,
            wav2lip_audio_input,
        )
        print(
            "*********************  # step-4 Adjust the audio video length Completed ***********************"
        )
    start_time = time.time()
    # Step-5 LipSync using wav2lip
    if audio_length_adjustment:
        if crop_supported:
            cropped_from = wav2lip_video_input
            crop_video(wav2lip_video_input)
            lipsync_video(wav2lip_audio_input, cropped_video, name_en, model)
        else:
            lipsync_video(wav2lip_audio_input, wav2lip_video_input, name_en, model)
    else:
        if crop_supported:
            cropped_from = input_video
            crop_video(input_video)
            lipsync_video(elevenlabs_audio_adjusted_wav, cropped_video, name_en, model)
        else:
            lipsync_video(elevenlabs_audio_adjusted_wav, input_video, name_en, model)
    end_time = time.time()
    print(f"Time taken in video generaion from coqui-tts {end_time - start_time}")

    # Step-6 Stitch the video back
    video_stitching(cropped_from, crop_supported, name, name_en)


@app.post("/api/synthesize/video")
async def get_video(request: VideoRequest = None):
    # Construct the file path
    if check_object_exists(f"user-lipsync-video/{request.name_en}.mp4"):
        try:
            response = get_video_from_minio(request.name_en)
        except Exception as e:
            return {"error": str(e)}

        # Set up the StreamingResponse
        video_response = StreamingResponse(
            response.stream(32 * 1024),  # Stream in chunks of 32 KB
            media_type="video/mp4",
        )

        # video_response.headers["Content-Disposition"] = f"inline; filename={request.name}.mp4"
        return video_response
    else:
        main(
            name=request.name,
            name_en=request.name_en,
            audio_length_adjustment=request.audio_length_adjustment,
            crop_supported=request.crop_supported,
            tts_suffix=request.tts_suffix,
            temperature=request.temperature,
            length_penalty=request.length_penalty,
            repetition_penalty=request.repetition_penalty,
            top_k=request.top_k,
            top_p=request.top_p,
            sentence_split=request.sentence_split,
            coqui_tts_api=request.coqui_tts_api
        )

        video_file = open(os.path.join(base_dir, "final_result.mp4"), mode="rb")
        print("video file location", os.path.join(base_dir, "final_result.mp4"))
        video_response = StreamingResponse(video_file, media_type="video/mp4")

        # return JSONResponse(
        #     content={"status_code": 200, "url":"https://himanshupersonals3.s3.amazonaws.com/final_result.mp4"},
        #     media_type="application/json"
        # ), video_response
        return video_response
        # return {"status_code": 200, "url":"https://himanshupersonals3.s3.amazonaws.com/final_result.mp4"}


@app.post("/api/v1/video/synthesize")
async def synthesize_video(request: VideoRequest = None):
    if check_object_exists(f"user-lipsync-video/{request.name}.mp4"):
        pass
    else:
        main(
            name=request.name,
            name_en=request.name_en,
            audio_length_adjustment=request.audio_length_adjustment,
            crop_supported=request.crop_supported,
            tts_suffix=request.tts_suffix,
            temperature=request.temperature,
            length_penalty=request.length_penalty,
            repetition_penalty=request.repetition_penalty,
            top_k=request.top_k,
            top_p=request.top_p,
            sentence_split=request.sentence_split,
            coqui_tts_api=request.coqui_tts_api
        )
    return {
        "status_code": 200,
        "status_message": "success",
        "url": f"https://cdn-api.dev.ks.samagra.io/cm-user-videos/user-lipsync-video/{request.name_en}.mp4",
    }

@app.post("/api/v1/video/prepare")
def prepare_video(request: prepare_video_request = None):
    start_time = request.start_time
    end_time = request.end_time
    input_video = "input_video.mp4"
    original = "original.mp4"
    input_video_compressed = "input_video_compressed.mp4"
    # step-1 compress the video
    compress_video(original, input_video_compressed, request.compression_ratio)
    print(
        "*********************  step-1 compress the video Completed  ***********************"
    )


    # step-2 cut the video in 3 parts
    cut_video(
        input_video_path=input_video_compressed,
        start_time=start_time,  # 0
        end_time=end_time,  # 7000ms
        output_part1="static_part_1.mp4",
        output_part2=input_video,
        output_part3="static_part_3.mp4",
    )
    print(
        "*********************  step-2 cut the video in 3 parts Completed  ***********************"
    )
    return {"status_code": 200}

@app.get("/health")
def health_check():
    return {"status_code": 200}
