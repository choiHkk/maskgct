## MaskGCT(Unofficial Implementation)
Unofficial Implementation of MaskGCT
This is an unofficial implementation of the [MaskGCT](https://arxiv.org/abs/2409.00750) model. Below is a detailed explanation of the approach taken in this project, including key differences from the original implementation and the scope of what is shared publicly.

### Samples
### Comparison: MaskGCT, Ditto-TTS, CLAM-TTS, Mega-TTS
* [zeroshot](./samples/zero_shot_tts)
* [celeb](./samples/celebrities)
* [anime](./samples/anime)

### Description
In this implementation:
* **No Semantic Tokens**: Unlike the original MaskGCT, this version does not utilize semantic tokens. As a result, this implementation does not support voice conversion functionality.
* **Custom Audio Tokenizer**: I trained an audio tokenizer from scratch using [Stable Audio Tools](https://github.com/stability-ai/stable-audio-tools) instead of relying on pre-existing tokenizers. (44100hz, 2048hop)
* **Text Tokenizer**: During training and inference, the model utilizes only the [**meta-llama/Llama-3.1-8B-Instruct**](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) tokenizer for input processing across all trained languages, without relying on phonemes or text normalization.
* **Data Preprocessing**: The preprocessing code is not publicly available. The dataset is structured using either Parquet or Arrow formats, with each item containing the fields "text", "codes", and "target_length".
* **Training Code**: The training codebase was solo-developed by me, drawing architectural inspiration from [Parler TTS](https://github.com/huggingface/parler-tts) and [Amphion](https://github.com/open-mmlab/Amphion) to align with my specific use case.
* **Dataset and Model Weights**: This implementation was trained on the [**Emilia**](https://huggingface.co/datasets/amphion/Emilia-Dataset), [**Libriheavy**](https://github.com/k2-fsa/libriheavy), [**HiFi-TTS**](https://huggingface.co/datasets/MikhailT/hifi-tts), [**MLS**](facebook/multilingual_librispeech), and **AIHub** datasets, which collectively include 11 languages. However, the resulting model weights are kept private and are not shared in this repository.
* **Open-Source Code**: Only the training and inference code are made publicly available to allow others to understand the implementation and adapt it to their own datasets.

This project aims to provide a functional recreation of MaskGCT’s core ideas while making some deliberate design choices tailored to our specific needs.

### Limitations
Since the dataset and model weights are not disclosed, users will need to prepare their own data and train the model themselves to use this implementation effectively.

### Inference
```python
import torch
import torchaudio
from transformers import AutoTokenizer

from maskgct_t2s import MaskGCT_T2S
from maskgct_s2a import MaskGCT_S2A
from dac_wrapper import DACModel

device = "cuda:0"
dtype = torch.float16

audio_encoder = DACModel.from_pretrained("parler-tts/dac_44khZ_8kbps").to(
    device=device, dtype=dtype
)
prompt_tokenizer = AutoTokenizer.from_pretrained("/path/to/model_t2s_dir")
model_t2s = MaskGCT_T2S.from_pretrained("/path/to/model_t2s_dir").to(
    device=device, dtype=dtype
)
model_s2a = MaskGCT_S2A.from_pretrained("/path/to/model_s2a_dir").to(
    device=device, dtype=dtype
)
```

```python
def get_target_length(
    prompt_text: str,
    input_text: str,
    prompt_codes_length: int,
    speed: float = 1.0,
):
    prompt_text_len = len(prompt_text.encode("utf-8"))
    input_text_len = len(input_text.encode("utf-8"))
    return int(prompt_codes_length / prompt_text_len * input_text_len / speed)

use_prompt = True

input_text = "Hello, my name is Hyunkyu Choi. I'm an AI researcher."

if use_prompt:
    prompt_audio_path = "/path/to/prompt_audio.wav"
    prompt_text = "This is a test prompt." # should be aligned with the prompt audio
    text = f"{prompt_text} {input_text}"
    prompt_input_ids = prompt_tokenizer(text, return_tensors="pt").input_ids.to(device=device)
    
    prompt_audio, sampling_rate = torchaudio.load(prompt_audio_path)
    if sampling_rate != 44100:
        prompt_audio = torchaudio.functional.resample(
            prompt_audio,
            sampling_rate,
            44100,
            # to prevent aliasing, https://pytorch.org/audio/stable/tutorials/audio_resampling_tutorial.html
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
        sampling_rate = 44100
    
    with torch.no_grad():
        prompt_audio = prompt_audio.unsqueeze(1).to(device=device, dtype=dtype)
        prompt_codes = audio_encoder.encode(prompt_audio, sample_rate=sampling_rate)
    target_length = get_target_length(prompt_text, input_text, prompt_codes.size(-1))
    
else:
    prompt_input_ids = prompt_tokenizer(input_text, return_tensors="pt").input_ids.to(device=device)
    prompt_codes = torch.full((1,9,1), 1024, device=device, dtype=torch.long)
    target_length = int(prompt_input_ids.shape[-1] * 8) # x8 ~ x16  or specific length
```

```python
inputs_t2s = {
    "target_length": target_length,
    "prompt_input_ids": prompt_input_ids,
    "speech_input_ids": prompt_codes[0,:1],
    "n_timesteps": 50,
    "cfg": 1.0,
    "rescale_cfg": 1.0,
    "temperature": 0.5,
    "filter_thres": 0.25,
}
with torch.no_grad():
    codes_0 = model_t2s.generate(**inputs_t2s) # [1, T_codes]

inputs_s2a = {
    "text_prompt_input_ids": prompt_input_ids,
    "speech_prompt_input_ids": prompt_codes.transpose(1, 2) if use_prompt else prompt_codes.transpose(1, 2), 
    "speech_input_ids": codes_0,
    "n_timesteps": [50, 10, 1, 1, 1, 1, 1, 1, 1],
    "cfg": 1.0,
    "rescale_cfg": 1.0,
    "temperature": 0.5, 
    "filter_thres": 0.25, 
}
with torch.no_grad():
    codes_0_8 = model_s2a.generate(**inputs_s2a) # [1, T_codes, n_q]
    audio_hat = audio_encoder.decode(codes_0_8.transpose(1, 2)) # [1, 1, T_audio]
    audio_hat = audio_hat.squeeze(1).detach().cpu() # [1, T_audio]

torchaudio.save("audio_hat.wav", audio_hat, 44100)
```

