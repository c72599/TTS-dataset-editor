import os
import torch
import argparse
import importlib
import tqdm

from opencc import OpenCC
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import warnings
warnings.filterwarnings("ignore")

"""
    Model Init
"""
cc = OpenCC('s2tw')

device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model_id = "openai/whisper-large-v3"
use_flash_attention = importlib.util.find_spec("flash-attn")

model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
    use_flash_attention_2=use_flash_attention)
model.to(device)

processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
)


"""
    Main Function
"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, help="path of dir of audio after vad preprocess")
    args = parser.parse_args()

    # Initialization
    dataset_root = args.dataset_root
    vad_result_path = os.path.join(dataset_root, "vad_result.txt")
    whisper_result_path = os.path.join(dataset_root, "whisper_v3_result.txt")

    # Check if this is a new preprocess or continue of previous preprocess
    skip = None
    if not os.path.exists(whisper_result_path):
        sf = open(whisper_result_path, 'w', encoding='utf-8')
        sf.write("FileName || Index || StartTime[in ms] || EndTime[in ms] || Transcript\n")
        sf.close()
    else:
        with open(whisper_result_path, 'r', encoding='utf-8') as rf:
            skip = rf.readlines()[-1].split('||')

    # Read all VAD result
    with open(vad_result_path, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()

    # Start preprocess
    pbar = tqdm.tqdm(lines[1:])
    for line in pbar:
        dir_name, file_name, _, _ = line.split('||')

        # Skip till where the preprocess stopped if this is the continue of previous preprocess
        if skip is not None:
            if dir_name == skip[0] and file_name == skip[1]:
                skip = None
            continue

        # Inference of the model
        audio_path = os.path.join(dataset_root, dir_name, f"{file_name}.mp3")
        result = pipe(audio_path, generate_kwargs={"task": "transcribe"})['text']
        result = cc.convert(result)

        # Save the result
        sf = open(whisper_result_path, 'a+', encoding='utf-8')
        sf.write(f"{line.rstrip()}||{result}\n")
        sf.close()