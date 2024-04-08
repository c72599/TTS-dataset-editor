import os
import pandas as pd
from multiprocessing import Pool
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

from .audio import audio_preprocess, log_melspectrogram, melspec_hparams


def read_ori_data(dataset_path):
    trans_path = os.path.join(dataset_path, "punctuation_restore_result.txt")
    assert os.path.exists(trans_path), "punctuation_restore_result.txt not exist"

    ori_data = []
    # read txt
    with open(trans_path, 'r', encoding="utf-8") as f:
        lines = f.readlines()[1:]
    # collect line
    for line in lines:
        file_name, vad_index, _, _, transcript = line.rstrip().split('||')
        ori_data.append((file_name, vad_index, transcript))

    return ori_data


def preprocess_dataset(single_ori_data):
    dataset_path, file_name, vad_index, transcript = single_ori_data

    abs_wav_path = os.path.join(dataset_path, file_name, f"{vad_index}.mp3")
    audio_segment = AudioSegment.from_mp3(abs_wav_path)
    audio_segment = audio_preprocess(audio_segment, melspec_hparams)
    melspec = log_melspectrogram(audio_segment, melspec_hparams)

    trim_head = detect_leading_silence(audio_segment, silence_threshold=-25) // 11.6
    trim_tail = detect_leading_silence(audio_segment.reverse(), silence_threshold=-25) // 11.6
    detect_start = max(trim_head - 10, 0)
    detect_end = min(melspec.shape[1] - trim_tail + 10, melspec.shape[1])

    return [[file_name], [vad_index], [[detect_start, detect_end]], [transcript], [True]]


def read_dataset(dataset_path):
    csv_path = os.path.join(dataset_path, "dataset_contents.csv")

    """
    new csv if csv not exist
    """
    if not os.path.exists(csv_path):
        ori_data = read_ori_data(dataset_path)
        ori_data = [(dataset_path, file_name, index, transcript) for file_name, index, transcript in ori_data]

        with Pool(6) as p:
            preprocess_data = p.map(preprocess_dataset, ori_data)

        file_names, vad_indexes, segments, transcripts, availables = [], [], [], [], []
        for file_name, vad_index, segment, transcript, available in sorted(preprocess_data):
            file_names.append(file_name)
            vad_indexes.append(vad_index)
            segments.append(segment)
            transcripts.append(transcript)
            availables.append(available)

        dataframe = pd.DataFrame({
            "FileName": file_names,
            "VadIndex": vad_indexes,
            "WhisperResult": transcripts,
            "Segments": segments,
            "Transcripts": transcripts,
            "Availables": availables
        })
        dataframe.to_csv(csv_path, index_label="Index", encoding="utf-8")

    dataframe = pd.read_csv(csv_path, encoding="utf-8")
    file_names = [eval(file_name) for file_name in dataframe["FileName"]]
    vad_indexes = [eval(index) for index in dataframe["VadIndex"]]
    whisper_results = [eval(whisper_result) for whisper_result in dataframe["WhisperResult"]]
    segments = [eval(segment) for segment in dataframe["Segments"]]
    transcripts = [eval(transcript) for transcript in dataframe["Transcripts"]]
    availables = [eval(available) for available in dataframe["Availables"]]

    return file_names, vad_indexes, whisper_results, segments, transcripts, availables