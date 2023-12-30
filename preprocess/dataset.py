import os
import pandas as pd
from multiprocessing import Pool
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from .audio import audio_preprocess, log_melspectrogram, melspec_hparams


def read_ori_data(dataset_path):
    trans_path = os.path.join(dataset_path, "trans_p_restored.txt")
    assert os.path.exists(trans_path), "trans_p_restored.txt not exist"

    ori_data = []
    # read txt
    with open(trans_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()[1:]
    # collect line
    for line in lines:
        wav_path, _, p_restore = line.rstrip().split('|')
        ori_data.append((wav_path, p_restore))

    return ori_data


def preprocess_dataset(single_ori_data):
    dataset_path, wav_path, transcript = single_ori_data

    abs_wav_path = os.path.join(dataset_path, wav_path)
    audio_segment = AudioSegment.from_mp3(abs_wav_path)
    audio_segment = audio_preprocess(audio_segment, melspec_hparams)
    melspec = log_melspectrogram(audio_segment, melspec_hparams)

    trim_head = detect_leading_silence(audio_segment, silence_threshold=-25) // 11.6
    trim_tail = detect_leading_silence(audio_segment.reverse(), silence_threshold=-25) // 11.6
    detect_start = max(trim_head - 10, 0)
    detect_end = min(melspec.shape[1] - trim_tail + 10, melspec.shape[1])

    return [[wav_path], [[detect_start, detect_end]], [transcript], [True]]


def read_dataset(dataset_path):
    csv_path = os.path.join(dataset_path, "processed_transcript.csv")

    """
    new csv if csv not exist
    """
    if not os.path.exists(csv_path):
        ori_data = read_ori_data(dataset_path)
        ori_data = [(dataset_path, wav_path, transcript) for wav_path, transcript in ori_data]

        with Pool(6) as p:
            preprocess_data = p.map(preprocess_dataset, ori_data)

        wav_paths, segments, transcripts, wav_types = [], [], [], []
        for wav_path, segment, transcript, wav_type in preprocess_data:
            wav_paths.append(wav_path)
            segments.append(segment)
            transcripts.append(transcript)
            wav_types.append(wav_type)

        dataframe = pd.DataFrame({
            "Path": wav_paths,
            "Segments": segments,
            "Whisper_res": transcripts,
            "Transcripts": transcripts,
            "Type": wav_types
        })
        dataframe.to_csv(csv_path, index_label="Index", encoding="utf-8")

    dataframe = pd.read_csv(csv_path, encoding="utf-8")
    paths = [eval(path) for path in dataframe["Path"]]
    segments = [eval(segment) for segment in dataframe["Segments"]]
    whisper_res = [eval(whisper_re) for whisper_re in dataframe["Whisper_res"]]
    transcripts = [eval(transcript) for transcript in dataframe["Transcripts"]]
    types = [eval(type) for type in dataframe["Type"]]

    return paths, segments, whisper_res, transcripts, types