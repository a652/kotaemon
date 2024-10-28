import cv2
import time
import numpy as np
from sklearn.cluster import KMeans
import os
from decouple import config

from theflow.settings import settings as flowsettings
from moviepy.editor import VideoFileClip

import openai
from pydub import AudioSegment
import os
import codecs
import tempfile

video_cache_dir: str = getattr(flowsettings, "KH_VIDEO_CACHE_DIR", None)

from pypinyin import pinyin, Style

def chinese_to_pinyin(text):
    # 将中文转换为拼音
    return '_'.join([item[0] for item in pinyin(text, style=Style.NORMAL)])

def preprocess_video(video_path):
    """
    预处理视频,提取所有帧
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("视频读取结束")
            break
        frames.append(frame)  # 保存原始彩色帧
    cap.release()
    print("视频预处理完成，共提取 {} 帧".format(len(frames)))
    return frames

def detect_shot_boundaries(frames, threshold=10):
    """
    使用帧差法检测镜头边界
    """
    shot_boundaries = []
    for i in range(1, len(frames)):
        prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diff = np.mean(np.abs(curr_frame.astype(int) - prev_frame.astype(int)))
        if diff > threshold:
            shot_boundaries.append(i)
    return shot_boundaries

def extract_keyframes(frames, shot_boundaries):
    """
    从每个镜头中提取关键帧
    """
    keyframes = []
    for i in range(len(shot_boundaries)):
        start = shot_boundaries[i-1] if i > 0 else 0
        end = shot_boundaries[i]
        shot_frames = frames[start:end]
        
        # 使用 K-means 聚类选择关键帧
        frame_features = np.array([cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).flatten() for frame in shot_frames])
        kmeans = KMeans(n_clusters=1, random_state=0).fit(frame_features)
        center_idx = np.argmin(np.sum((frame_features - kmeans.cluster_centers_[0])**2, axis=1))
        
        keyframes.append(shot_frames[center_idx])
    
    return keyframes

def save_keyframes(keyframes, output_dir, filename) -> list[str]:
    """
    保存关键帧到指定目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    outputpaths = []
    
    for i, keyframe in enumerate(keyframes):
        output_path = os.path.join(output_dir, f'keyframe_{filename}_{i:04d}.png')
        cv2.imwrite(output_path, keyframe)
        outputpaths.append(output_path)
    
    print(f"已保存 {len(keyframes)} 个关键帧到 {output_dir}")
    return outputpaths

def extract_video_keyframes(filename, video_path, output_dir=video_cache_dir) -> list[str]:
    # filename = os.path.splitext(os.path.basename(video_path))[0]
    frames = preprocess_video(video_path)
    shot_boundaries = detect_shot_boundaries(frames)
    keyframes = extract_keyframes(frames, shot_boundaries)
    return save_keyframes(keyframes, output_dir, filename)

def extract_video_audio(filename, video_path, output_dir=video_cache_dir) -> str:
    # filename = os.path.splitext(os.path.basename(video_path))[0]
    # 加载视频文件
    clip = VideoFileClip(video_path)
    output_path = os.path.join(output_dir,f'audio_{filename}.mp3')
    # 提取音频并保存到新的文件
    clip.audio.write_audiofile(output_path)
    return output_path


def transcribe_audio_with_whisper(audio_file_path) -> str:
    """
    Transcribe an audio file using OpenAI's Whisper API.

    Args:
    - audio_file_path: Path to the audio file to transcribe.

    Returns:
    - The transcribed text as a string.
    """
    api_key = config("OPENAI_API_KEY", default="")
    # base_url = config("OPENAI_API_BASE", default="https://api.openai.com/v1")
    base_url = "https://ai-gateway.mininglamp.com/v1/"
    openai.api_key = api_key
    openai.base_url = base_url
    print(f"Transcribing {audio_file_path} with Whisper API...")
    with open(audio_file_path, "rb") as audio_file:
        for i in range(3):
            try:
                response = openai.audio.transcriptions.create(
                    language="zh",
                    model='whisper-1',
                    file=audio_file,
                )
                return response.text
            except Exception as e:
                print(f"Error transcribing audio: {e}, retrying...")
                time.sleep(5)

def split_and_transcribe_audio(filename, file_path, segment_length_seconds=30, output_dir=video_cache_dir) -> str:
    try:
        song = AudioSegment.from_file(file_path)
    except Exception as e:
        raise Exception(f"Error loading audio file: {e}")

    segment_length_ms = segment_length_seconds * 1000  # Correct calculation of milliseconds
    transcripts = []

    with tempfile.TemporaryDirectory() as temp_dir:
        for i, segment in enumerate([song[i:i+segment_length_ms] for i in range(0, len(song), segment_length_ms)]):
            segment_file_path = os.path.join(temp_dir, f"{filename}_segment_{i}.mp3")
            segment.export(segment_file_path, format="mp3")
            
            transcript = transcribe_audio_with_whisper(segment_file_path)
            time_in_seconds = i * segment_length_seconds
            timestamp = f"[{time_in_seconds // 60:02d}:{time_in_seconds % 60:02d}]"
            transcripts.append(timestamp + " " + transcript)

    output_path = os.path.join(output_dir,f'text_{filename}.txt')
    with codecs.open(output_path, 'w', encoding='utf-8') as f:  # Using UTF-8 encoding
        f.write("\n".join(transcripts))
    return "\n".join(transcripts)


if __name__ == "__main__":
    video_path = "/Users/zhangcheng/Downloads/测试集-第一版/哇哈哈.MP4"
    output_dir = video_cache_dir
    # extract_video_keyframes(video_path, output_dir)

    # 加载视频文件
    clip = VideoFileClip(video_path)
    # 提取音频并保存到新的文件
    clip.audio.write_audiofile(os.path.join(output_dir,'extracted_audio_wahaha.mp3'))
    # print(transcribe_audio_with_whisper('/Users/zhangcheng/code/python/kotaemon/ktem_app_data/video_cache_dir/converted_audio_wahaha.mp3'))
    print(transcribe_audio_with_whisper(os.path.join(output_dir,'extracted_audio_wahaha.mp3')))

