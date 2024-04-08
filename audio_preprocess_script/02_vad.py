import os
import argparse
import torch
import torchaudio
import tqdm

from pydub import AudioSegment

torchaudio.set_audio_backend("sox_io")
torch.set_num_threads(1)
SAMPLING_RATE = 16000

model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True,
                              onnx=False)

(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


"""
    Main Function
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, help="path of dir of audio dataset")
    args = parser.parse_args()

    # Initialization
    dataset_root = args.dataset_root
    save_root = f"{dataset_root}_vad"
    vad_result_path = os.path.join(save_root, "vad_result.txt")
    os.makedirs(save_root, exist_ok=True)

    # Check if this is a new preprocess or continue of previous preprocess
    skip = None
    if not os.path.exists(vad_result_path):
        sf = open(vad_result_path, 'w', encoding='utf-8')
        sf.write("FileName || Index || StartTime[in ms] || EndTime[in ms] || Transcript\n")
        sf.close()
    else:
        with open(vad_result_path, 'r', encoding='utf-8') as rf:
            skip = rf.readlines()[-1].split('||')

    # Collect all audio file in dataset dir
    file_lists = []
    for root, dirs, files in os.walk(dataset_root):
        for file in files:
            if file[-3:] != 'wav':
                continue

            audio_path = os.path.join(root, file)
            file_name = file.split('_', 1)[1][:-13]
            file_lists.append((file_name, audio_path))
    file_lists.sort()

    # Start preprocess
    pbar = tqdm.tqdm(file_lists)
    for file_name, audio_path in pbar:

        # Skip the audio already processed
        if skip is not None and file_name != skip[0]:
            continue

        # Read audio
        save_dir = os.path.join(save_root, file_name)
        audio = AudioSegment.from_file(audio_path)
        audio = match_target_amplitude(audio, -20.0)
        audio.set_frame_rate(16000)

        # VAD
        wav = read_audio(audio_path, sampling_rate=SAMPLING_RATE)
        speech_timestamps = get_speech_timestamps(wav, model,
                                                  min_speech_duration_ms=1000, #500
                                                  min_silence_duration_ms=300, #300
                                                  sampling_rate=SAMPLING_RATE,
                                                  return_seconds=True)

        # Skip if there is no segment detected in VAD preprocess
        if not speech_timestamps:
            continue

        # Save the result of current audio
        os.makedirs(save_dir, exist_ok=True)
        for idx, segment in enumerate(speech_timestamps):

            # Skip the segment already processed
            if skip is not None:
                if f"{idx:0>4d}" == skip[1]:
                    skip = None
                continue

            # Save audio segment
            save_path = os.path.join(save_dir, f"{idx:0>4d}.mp3")
            sec_start, sec_end = segment['start'], segment['end']
            lp = int(max(0, sec_start * 1000 - 250))
            rp = int(min(len(audio), sec_end * 1000 + 250))
            save_audio = audio[lp: rp]
            save_audio.export(save_path)

            # Save the result
            with open(vad_result_path, 'a+', encoding='utf-8') as f:
                f.write(f"{file_name}||{idx:0>4d}||{lp}||{rp}\n")




