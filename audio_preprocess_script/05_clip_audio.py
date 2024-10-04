import sys
import os
import re
import argparse
import tqdm
from pydub import AudioSegment

root_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, root_folder_path)

from editor_function.dataset import read_dataset
from editor_function.audio import melspec_hparams, audio_preprocess, log_melspectrogram


def convert_timestemp(segment, mel_len, audio_len):
    sp, ep = segment
    c_sp = int((sp / (mel_len-1)) * audio_len)
    c_ep = int((ep / (mel_len-1)) * audio_len)

    return c_sp, c_ep


"""
    Main Function
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, help="path of dir of dataset")
    parser.add_argument("--task", type=str, help="filter by audio task")
    args = parser.parse_args()

    # Initialization
    dataset_root = args.dataset_root
    export_dir_path = os.path.join(dataset_root, "export")
    export_transcript_path = os.path.join(export_dir_path, f"transcript.txt")
    os.makedirs(export_dir_path, exist_ok=True)

    # Load dataset content
    dataset_contents = read_dataset(dataset_root)

    pbar = tqdm.tqdm(range(len(dataset_contents['FileName'])))
    for findex in pbar:
        fname = dataset_contents['FileName'][findex][0]
        vadindex = dataset_contents['VadIndex'][findex][0]
        segments = dataset_contents['Segments'][findex]
        transcripts = dataset_contents['Transcripts'][findex]
        tasks = dataset_contents['Tasks'][findex]

        # Load audio
        fpath = os.path.join(dataset_root, fname, f"{vadindex}.mp3")
        audio = AudioSegment.from_mp3(fpath)
        audio = audio_preprocess(audio, melspec_hparams)
        audio_lenght = len(audio)
        melspec = log_melspectrogram(audio, melspec_hparams)
        melspec_length = melspec.shape[1]

        for cindex, (segment, transcript, task) in enumerate(zip(segments, transcripts, tasks)):
            if not args.task in task:
                continue

            # Set audio clip
            clip_name = f"{fname}_{vadindex}_{cindex}"
            clip_export_path = os.path.join(export_dir_path, f"{clip_name}.mp3")
            clip_sp, clip_ep = convert_timestemp(segment, melspec_length, audio_lenght)
            clip_audio = audio[clip_sp: clip_ep+1]

            # Save audio clip
            clip_audio.export(clip_export_path)
            with open(export_transcript_path, 'a+', encoding="utf-8") as f:
                f.write(f"{clip_name}|{transcript}\n")