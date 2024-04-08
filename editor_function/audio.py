import librosa
import numpy as np


melspec_hparams = {
    "max_wav_value": 32768.0,
    "sampling_rate": 22050,
    "filter_length": 1024,
    "hop_length": 256,
    "win_length": 1024,
    "n_mel_channels": 80,
    "mel_fmax": 8000.0
}


def match_target_amplitude(sound, target_dBFS):
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)


def audio_preprocess(audio_segment, hparams):
    # Normalize audio
    audio_segment = match_target_amplitude(audio_segment, -20.0)
    # Set framerate
    audio_segment = audio_segment.set_frame_rate(hparams["sampling_rate"])
    # Mono channel only
    audio_segment = audio_segment.set_channels(1)
    return audio_segment


def pydub_to_np(audio_segment):
    """
    Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels],
    where each value is in range [-1.0, 1.0].
    Returns tuple (audio_np_array, sample_rate).
    """
    return np.array(audio_segment.get_array_of_samples(), dtype=np.float32).reshape((audio_segment.channels, -1))


def log_melspectrogram(audio_segment, hparams):
    np_audio = pydub_to_np(audio_segment)
    melspec = librosa.feature.melspectrogram(y=np_audio,
                                             sr=hparams["sampling_rate"],
                                             n_fft=hparams["filter_length"],
                                             hop_length=hparams["hop_length"],
                                             win_length=hparams["win_length"],
                                             n_mels=hparams["n_mel_channels"],
                                             window='hann',
                                             fmax=hparams["mel_fmax"])
    logmelspec = librosa.power_to_db(melspec)[0]
    return logmelspec