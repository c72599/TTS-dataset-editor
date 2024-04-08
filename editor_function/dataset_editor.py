import os
import pandas as pd
import ipywidgets as widgets
from pydub import AudioSegment

from editor_function.dataset import read_dataset
from editor_function.audio import audio_preprocess, log_melspectrogram, melspec_hparams


class DatasetEditor():
    def __init__(self, ipd, plt):
        self.dataset_path = ""
        self.ipd = ipd
        self.plt = plt

        # dataset variable
        self.file_names = None
        self.indexes = None
        self.whisper_results = None
        self.segments = None
        self.transcripts = None
        self.availables = None

        # ui variable
        self.index = None
        self.dataset_path = None
        self.audio = None
        self.melspec = None
        self.layout_segments_group = None

        # key: btn_del, value: all obj in the same segment
        self.dict_segment = {}
        # record all segments layout
        self.layout_segments_group = []

        # control ui init
        btn_add = widgets.Button(description="新增片段")
        btn_add.on_click(self.btn_add_segment)
        btn_prev = widgets.Button(description="前一個音檔")
        btn_prev.on_click(self.prev_wav)
        btn_next = widgets.Button(description="下一個音檔")
        btn_next.on_click(self.next_wav)
        btn_del = widgets.Button(description="刪除此音檔")
        btn_del.on_click(self.del_audio)
        btn_save = widgets.Button(description="儲存CSV")
        btn_save.on_click(self.save_csv)
        self.control_bar = widgets.HBox([btn_add, btn_prev, btn_next, btn_del, btn_save])

    def load_dataset(self, dataset_path):
        self.dataset_path = dataset_path

        file_names, indexes, whisper_results, segments, transcripts, availables = read_dataset(dataset_path)
        self.file_names = file_names
        self.indexes = indexes
        self.whisper_results = whisper_results
        self.segments = segments
        self.transcripts = transcripts
        self.availables = availables

    def set_index(self, index):
        self.index = index

    def refresh_display(self, change_wav):
        # clear display
        self.ipd.clear_output()

        print(f"原檔名稱：{self.file_names[self.index][0]}/{self.indexes[self.index][0]}.mp3")
        print(f"音檔編號：{self.index}，音檔狀態：{self.availables[self.index][0]}")

        # display melspec
        if self.availables[self.index][0]:
            # change the audio => reload audio
            if change_wav:
                self.layout_segments_group = []
                audio_path = os.path.join(self.dataset_path, self.file_names[self.index][0], f"{self.indexes[self.index][0]}.mp3")
                self.audio = self.ipd.Audio(audio_path)

                audio_segment = AudioSegment.from_mp3(audio_path)
                audio_segment = audio_preprocess(audio_segment, melspec_hparams)
                self.melspec = log_melspectrogram(audio_segment, melspec_hparams)[::-1, :]

                for segment, transcript in zip(self.segments[self.index], self.transcripts[self.index]):
                    start, end = segment
                    self.add_segment(start, end, transcript)

            # paint melspec
            self.plt.close()
            new_melspec = self.melspec.copy()
            for layout_segment in self.layout_segments_group:
                _, layout_slider = layout_segment.children
                start, end, _ = layout_slider.children
                start, end = start.value, end.value
                if start < end:
                    new_melspec[:, start: end] += 60
            self.plt.xticks(range(0, self.melspec.shape[1], 50))
            self.plt.imshow(new_melspec, interpolation='nearest')
            self.ipd.display(self.audio)

        # display control
        control_pannel = widgets.VBox([self.control_bar, widgets.VBox(self.layout_segments_group)])
        self.ipd.display(control_pannel)

    def add_segment(self, start=0, end=0, transcript=None):
        # text transcript
        if not transcript:
            transcript = self.whisper_results[self.index][0]
        textbox = widgets.Text(value=transcript, description='文字', layout=widgets.Layout(width='80%'))
        # slider start
        slider_start = widgets.IntSlider(value=start, description="起始", max=self.melspec.shape[1] - 1)
        # slider end
        slider_end = widgets.IntSlider(value=end, description="結束", max=self.melspec.shape[1] - 1)
        # btn delete
        btn_del = widgets.Button(description="刪除片段")

        segment_layout_list = [slider_start, slider_end, btn_del]
        layout_slide = widgets.HBox(segment_layout_list)
        layout_segment = widgets.VBox([textbox, layout_slide])
        self.layout_segments_group.append(layout_segment)
        self.dict_segment[btn_del] = layout_segment

        btn_del.on_click(self.btn_del_segment)
        slider_start.observe(self.slider_slide)
        slider_end.observe(self.slider_slide)

    def btn_add_segment(self, button):
        self.add_segment()
        self.refresh_display(change_wav=False)

    def btn_del_segment(self, button):
        stack = [self.dict_segment[button]]
        self.layout_segments_group.remove(self.dict_segment[button])
        del self.dict_segment[button]

        while len(stack) > 0:
            current = stack.pop()
            if hasattr(current, 'children'):
                for child in current.children:
                    stack.append(child)
            current.close()
            del current

    def slider_slide(self, change):
        new_melspec = self.melspec.copy()
        for layout_segment in self.layout_segments_group:
            _, layout_slider = layout_segment.children
            start, end, _ = layout_slider.children
            start, end = start.value, end.value
            if start < end:
                new_melspec[:, start: end] += 60
        self.plt.imshow(new_melspec, interpolation='nearest')
        self.plt.gcf().canvas.draw()

    def save_segment(self):
        self.segments[self.index] = []
        self.transcripts[self.index] = []

        for layout_segment in self.layout_segments_group:
            textbox, layout_slider = layout_segment.children
            start, end, _ = layout_slider.children
            self.segments[self.index].append((start.value, end.value))
            self.transcripts[self.index].append(textbox.value)

    def prev_wav(self, button):
        self.save_segment()
        if self.index > 0:
            self.index -= 1
        while self.index > 0 and not self.availables[self.index][0]:
            self.index -= 1
        self.refresh_display(change_wav=True)

    def next_wav(self, button):
        self.save_segment()
        if self.index < len(self.availables) - 1:
            self.index += 1
        while self.index < len(self.availables) - 1 and not self.availables[self.index][0]:
            self.index += 1
        self.refresh_display(change_wav=True)

    def del_audio(self, button):
        self.availables[self.index][0] = not self.availables[self.index][0]
        self.refresh_display(change_wav=True)

    def save_csv(self, button):
        self.save_segment()
        dataframe = pd.DataFrame({
            "FileName": self.file_names,
            "Index": self.indexes,
            "WhisperResult": self.whisper_results,
            "Segments": self.segments,
            "Transcripts": self.transcripts,
            "Availables": self.availables
        })
        dataframe.to_csv(os.path.join(self.dataset_path, "dataset_contents.csv"), index_label="Index", encoding="utf-8")

