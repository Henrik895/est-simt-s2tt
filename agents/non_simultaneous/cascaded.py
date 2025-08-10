from simuleval.agents import AgentPipeline, SpeechToTextAgent, TextToTextAgent
from simuleval.agents import ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from typing import Optional, Dict
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

import numpy as np
import torch
import whisper


LANG_ET = "et"
LANG_EN = "en"
LANG_RU = "ru"
ALLOWED_LANGUAGES = [LANG_ET, LANG_EN, LANG_RU]

WHISPER_TINY = "tiny"
WHISPER_BASE = "base"
WHISPER_SMALL = "small"
WHISPER_MEDIUM = "medium"
WHISPER_LARGE = "large"
WHISPER_TURBO = "turbo"
WHISPER_SIZES = [
    WHISPER_TINY,
    WHISPER_BASE,
    WHISPER_SMALL,
    WHISPER_MEDIUM,
    WHISPER_LARGE,
    WHISPER_TURBO,
]

LANG_TO_NLLB_TOKEN: Dict[str, str] = {
    LANG_ET: "est_Latn",
    LANG_EN: "eng_Latn",
    LANG_RU: "rus_Cyrl",
}

NLLB_CHECKPOINT = "facebook/nllb-200-distilled-1.3B"


class WhisperAgent(SpeechToTextAgent):
    def __init__(self, args = None):
        super().__init__(args)
        self.source_language = args.source_language
        self.device = args.device
        self.fp16_enabled = torch.cuda.is_available()
        self.model = whisper.load_model(args.whisper_model, device=self.device)
        print("Cascaded: WhisperAgent")
        print("Source language:", self.source_language)
        print("Device:", self.device)
        print("FP16:", self.fp16_enabled)
        print("Whisper:", args.whisper_model)
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--source-language", default=LANG_EN, choices=ALLOWED_LANGUAGES)
        parser.add_argument("--whisper-model", default=WHISPER_TURBO, type=str, choices=WHISPER_SIZES)

    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states

        # Wait for the entire source sequence to finish before
        # proceeding with the NLLB-200 translation.
        if not states.source_finished:
            return ReadAction()

        options = whisper.DecodingOptions(
            language=self.source_language,
            without_timestamps=True,
            fp16=self.fp16_enabled,
            temperature=0,
            beam_size=5,
            sample_len=250,
        )

        audio_type = "float16" if self.fp16_enabled else "float32"
        audio = whisper.pad_or_trim(np.array(states.source).astype(audio_type))
        mel = whisper.log_mel_spectrogram(audio, self.model.dims.n_mels).to(self.model.device)
        output = self.model.decode(mel, options)
        prediction = output.text
        
        return WriteAction(prediction, finished=states.source_finished)


class NLLBAgent(TextToTextAgent):
    def __init__(self, args = None):
        super().__init__(args)
        self.source_language = args.source_language
        self.target_language = args.target_language
        self.device = args.device
        self.fp16_enabled = torch.cuda.is_available()
        self.audio_type = torch.float16 if self.fp16_enabled else torch.float32
        print("Device:", self.device)
        print("FP16:", self.fp16_enabled)
        print("Audio type:", self.audio_type)
        print("NLLB:", NLLB_CHECKPOINT)
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(NLLB_CHECKPOINT)
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_CHECKPOINT, torch_dtype=self.audio_type).to(self.device)
        self.nllb_tokenizer.src_lang = LANG_TO_NLLB_TOKEN[self.source_language]
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--source-language", default=LANG_EN, type=str, choices=ALLOWED_LANGUAGES)
        parser.add_argument("--target-language", default=LANG_ET, type=str, choices=ALLOWED_LANGUAGES)

    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states
        if not states.source_finished:
            return ReadAction()

        tgt_lang_id = self.nllb_tokenizer.convert_tokens_to_ids(LANG_TO_NLLB_TOKEN[self.target_language])
        # 2 - </s> token is required for generating a new sentence
        translation_decoder_ids = torch.tensor([[2, tgt_lang_id]]).to(self.device)
        # states.source is an array and the last element is the full transcription
        translation_inputs = self.nllb_tokenizer(text=states.source[-1], return_tensors="pt").to(self.device)

        tokens_generated = 0 # Currently, the limit is set to 250 to avoid accidentally long generations
        while True:
            with torch.no_grad():
                output = self.nllb_model(input_ids=translation_inputs["input_ids"], decoder_input_ids=translation_decoder_ids)

            # Regular greedy decoding is used
            next_token_logits = output.logits[:, -1, :]
            next_token_scores = next_token_logits.softmax(dim=1)

            next_token = next_token_scores.argmax().unsqueeze(0).unsqueeze(0)

            translation_decoder_ids = torch.cat((translation_decoder_ids, next_token), dim=1).to(self.device)

            if (translation_decoder_ids.squeeze()[-1] == self.nllb_tokenizer.eos_token_id):
                break

            tokens_generated += 1

            if tokens_generated >= 250:
                if translation_decoder_ids.squeeze()[-1] != self.nllb_tokenizer.eos_token_id:
                    eos_token = torch.tensor([[self.nllb_tokenizer.eos_token_id]]).to(self.device)
                    translation_decoder_ids = torch.cat((translation_decoder_ids, eos_token), dim=1).to(self.device)
                break
        
        translated_text = self.nllb_tokenizer.decode(translation_decoder_ids[0], skip_special_tokens=True)
        
        return WriteAction(translated_text, finished=states.source_finished)

@entrypoint
class CascadedPipeline(AgentPipeline):
    pipeline = [
        WhisperAgent,
        NLLBAgent,
    ]
