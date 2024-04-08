# TTS-dataset-editor
TTS資料集編輯的程式，處理自己蒐集的聲音資料集

---
## Preprocess script

### _**/ audio_preprocess_script / 01_segmentation.py**_ <br>
利用 **ffmpeg** 將過長的音檔切成固定長度的數段子音檔 <br>

* 輸入參數 root_of_datasets： 數個資料集的根目錄<br>

資料集根目錄 <br>
&emsp;&emsp; ├─ 資料集 A <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 音檔 A.wav <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 音檔 B.wav <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 音檔 C.wav <br>
&emsp;&emsp; │ <br>
&emsp;&emsp; ├─ 資料集 B <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 音檔 D.wav <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 音檔 E.wav <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 音檔 F.wav <br>

* 輸入參數 length： 想要分割的長度 (以秒為單位)<br> 

---
### _**/ audio_preprocess_script / 02_vad.py**_ <br>
利用 **snakers4/silero-vad** 將音檔進行切音處理 <br> 

* 輸入參數 dataset_root： 資料集目錄<br>

資料集目錄 <br>
&emsp;&emsp; ├─ 音檔 A.wav <br>
&emsp;&emsp; ├─ 音檔 B.wav <br>
&emsp;&emsp; ├─ 音檔 C.wav <br>

---
### _**/ audio_preprocess_script / 03_whisper_v3.py**_ <br>
利用 **openai/whisper-large-v3** 將音檔進行語音文字辨識 <br>

* 輸入參數 dataset_root： VAD 處理後的資料集目錄<br>

資料集目錄 <br>
&emsp;&emsp; ├─ 音檔 A <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 A.mp3 <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 B.mp3 <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 C.mp3 <br>
&emsp;&emsp; │ <br>
&emsp;&emsp; ├─ 音檔 B <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 D.mp3 <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 E.mp3 <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 F.mp3 <br>
&emsp;&emsp; │ <br>
&emsp;&emsp; └─ vad_reasult.txt

---
### _**/ audio_preprocess_script / 04_punctuation_restore.py**_ <br>
利用 **p208p2002/zh-wiki-punctuation-restore** 將 Whisper 辨識過後的文字補上標點符號 <br>

* 輸入參數 dataset_root： Whisper V3 處理後的資料集目錄<br>

資料集目錄 <br>
&emsp;&emsp; ├─ 音檔 A <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 A.mp3 <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 B.mp3 <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 C.mp3 <br>
&emsp;&emsp; │ <br>
&emsp;&emsp; ├─ 音檔 B <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 D.mp3 <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 E.mp3 <br>
&emsp;&emsp; │&emsp;&emsp; ├─ 片段 F.mp3 <br>
&emsp;&emsp; │ <br>
&emsp;&emsp; ├─ vad_reasult.txt <br>
&emsp;&emsp; └─ whisper_reasult.txt

---
## Editor UI
基於Jupyter notebook的資料集編輯介面

![image](image/dataset_editor_ui.PNG)
* 編號 1 ：顯示音檔檔名、在資料集中的音檔編號、語音檔是否可用
* 編號 2 ：顯示音檔轉為轉為 mel spectrogram 的結果
* 編號 3 ：播放整段音檔
* 編號 4 ：音檔編輯介面。可以為音檔新增/刪除片段、編輯片段時間、邊集片段文本內容與啟用/不啟用音檔
