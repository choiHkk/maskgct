import evaluate
import librosa
from faster_whisper import WhisperModel
from transformers import WhisperTokenizer


def transcribe(whisper, audio, sampling_rate):
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)

    transcription = ""
    try:
        segments, info = whisper.transcribe(audio, beam_size=5, vad_filter=True)
        language = info.language
        for segment in segments:
            transcription += segment.text
        transcription = transcription.strip()
    except:
        language = "en"

    return transcription, language


def wer(
    asr_model_name_or_path,
    texts,
    audios,
    device,
    sampling_rate,
):

    if device.index is None:
        device_index = 0
    else:
        device_index = device.index
    whisper = WhisperModel(
        asr_model_name_or_path, device_index=device_index, compute_type="float16"
    )
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    transcriptions = [transcribe(whisper, audio, sampling_rate) for audio in audios]

    tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
    english_normalizer = tokenizer.normalize
    basic_normalizer = tokenizer.basic_normalize

    normalized_predictions = []
    normalized_references = []

    for (transcription, language), ref in zip(transcriptions, texts):
        normalizer = english_normalizer if language == "en" else basic_normalizer
        norm_ref = normalizer(ref)
        if len(norm_ref) > 0:
            norm_pred = normalizer(transcription)
            normalized_predictions.append(norm_pred)
            normalized_references.append(norm_ref)

    word_error = 100 * metric_wer.compute(
        predictions=normalized_predictions, references=normalized_references
    )
    character_error = 100 * metric_cer.compute(
        predictions=normalized_predictions, references=normalized_references
    )

    return word_error, character_error, normalized_predictions
