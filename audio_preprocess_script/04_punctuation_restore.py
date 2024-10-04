import os
import re
import argparse
import tqdm
import torch

from zhpr.predict import DocumentDataset, merge_stride, decode_pred
from transformers import AutoModelForTokenClassification, AutoTokenizer
from torch.utils.data import DataLoader


class PunctuationRestore(object):
    def __init__(self, device="cpu"):
        self.device = device

        model_name = 'p208p2002/zh-wiki-punctuation-restore'
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict_step(self, batch):
        batch_out = []
        batch_input_ids = batch

        encodings = {'input_ids': batch_input_ids}
        output = self.model(**encodings)

        predicted_token_class_id_batch = output['logits'].argmax(-1)
        for predicted_token_class_ids, input_ids in zip(predicted_token_class_id_batch, batch_input_ids):
            out = []
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

            input_ids = input_ids.tolist()
            try:
                input_id_pad_start = input_ids.index(self.tokenizer.pad_token_id)
            except:
                input_id_pad_start = len(input_ids)
            input_ids = input_ids[:input_id_pad_start]
            tokens = tokens[:input_id_pad_start]

            predicted_tokens_classes = [self.model.config.id2label[t.item()] for t in predicted_token_class_ids]
            predicted_tokens_classes = predicted_tokens_classes[:input_id_pad_start]

            for token, ner in zip(tokens, predicted_tokens_classes):
                out.append((token, ner))
            batch_out.append(out)

        return batch_out

    def restore(self, text, window_size=256, step=200):
        en_words = set(re.findall("[a-zA-Z]+", text))
        text = text.replace(" ", "<s>")
        dataset = DocumentDataset(text.lower(), window_size=window_size, step=step)
        dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=5)

        model_pred_out = []
        for batch in dataloader:
            batch = batch.to(self.device)
            batch_out = self.predict_step(batch)
            for out in batch_out:
                model_pred_out.append(out)

        merge_pred_result = merge_stride(model_pred_out, step)
        merge_pred_result_deocde = decode_pred(merge_pred_result)
        merge_pred_result_deocde = ''.join(merge_pred_result_deocde)
        merge_pred_result_deocde = merge_pred_result_deocde.replace("[UNK]", "")
        merge_pred_result_deocde = merge_pred_result_deocde.replace("<s>", " ")
        for word in en_words:
            if word.lower() in merge_pred_result_deocde:
                merge_pred_result_deocde = merge_pred_result_deocde.replace(word.lower(), word)

        return merge_pred_result_deocde


"""
    Main Function
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, help="path of dir of audio after whisper preprocess")
    args = parser.parse_args()

    # Initialization
    dataset_root = args.dataset_root
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ps = PunctuationRestore(device=device)
    whisper_result_path = os.path.join(dataset_root, "whisper_v3_result.txt")
    punctuation_restore_result_path = os.path.join(dataset_root, "punctuation_restore_result.txt")

    # Check if this is a new preprocess or continue of previous preprocess
    skip = None
    if not os.path.exists(punctuation_restore_result_path):
        sf = open(punctuation_restore_result_path, 'w', encoding='utf-8')
        sf.write("FileName || Index || StartTime[in ms] || EndTime[in ms] || Transcript\n")
        sf.close()
    else:
        with open(punctuation_restore_result_path, 'r', encoding='utf-8') as rf:
            skip = rf.readlines()[-1].split('||')

    # Read all whisper result
    with open(whisper_result_path, 'r', encoding='utf-8') as rf:
        lines = rf.readlines()

    # Start preprocess
    pbar = tqdm.tqdm(lines[1:])
    for line in pbar:
        dir_name, file_name, start_time, end_time, transcript = line.rstrip().split('||')

        # Skip till where the preprocess stopped if this is the continue of previous preprocess
        if skip is not None:
            if dir_name == skip[0] and file_name == skip[1]:
                skip = None
            continue

        # Inference of the model
        p_restore = ps.restore(transcript)

        # Save the result
        sf = open(punctuation_restore_result_path, 'a+', encoding='utf-8')
        sf.write(f"{dir_name}||{file_name}||{start_time}||{end_time}||{p_restore}\n")
        sf.close()
