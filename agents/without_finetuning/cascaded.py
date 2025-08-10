from simuleval.agents import AgentPipeline, SpeechToTextAgent, TextToTextAgent
from simuleval.agents import ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.data.segments import TextSegment
from simuleval.utils import entrypoint
from typing import Optional, Dict
from examples.demo.silero_vad import SileroVADAgent
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


class WhisperStates(AgentStates):
    def __init__(self):
        super().__init__()
        self.whisper_segments = []

    def reset(self):
        super().reset()
        self.whisper_segments = []


class WhisperAgent(SpeechToTextAgent):
    def __init__(self, args = None):
        super().__init__(args)
        self.wait_k = args.wait_k
        self.source_language = args.source_language
        self.device = args.device
        self.fp16_enabled = torch.cuda.is_available()
        self.model = whisper.load_model(args.whisper_model, device=self.device)
        print("Cascaded: WhisperAgent")
        print("Source language:", self.source_language)
        print("Device:", self.device)
        print("FP16:", self.fp16_enabled)
        print("Whisper:", args.whisper_model)
        print("wait-k:", self.wait_k)

    def build_states(self):
        return WhisperStates()
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--wait-k", default=2, type=int)
        parser.add_argument("--source-language", default=LANG_EN, choices=ALLOWED_LANGUAGES)
        parser.add_argument("--whisper-model", default=WHISPER_TURBO, type=str, choices=WHISPER_SIZES)

    def policy(self, states: Optional[WhisperStates] = None):
        if states is None:
            states = self.states
        if states.source_sample_rate == 0:
            length_in_seconds = 0
        else:
            length_in_seconds = round(len(states.source) / states.source_sample_rate)

        if not states.source_finished and length_in_seconds < self.wait_k:
            return ReadAction()
        
        # It is important to not process empty audio segments because this causes the
        # whisper to generate random words.
        if len(states.source) == 0:
            if states.source_finished:
                return WriteAction("", finished=states.source_finished)
            return ReadAction()

        segment_contents = list(map(lambda x: x.content, self.states.whisper_segments))
        segment_contents = list(filter(lambda x: x != "", segment_contents)) # Remove empty segments

        if len(segment_contents) == 0:
            prefix = None
        else:
            prefix = " ".join(segment_contents)

        options = whisper.DecodingOptions(
            prefix=prefix,
            language=self.source_language,
            without_timestamps=True,
            fp16=self.fp16_enabled,
            temperature=0,
            beam_size=5,
            sample_len=30,
        )

        audio_type = "float16" if self.fp16_enabled else "float32"
        audio = whisper.pad_or_trim(np.array(states.source).astype(audio_type))
        mel = whisper.log_mel_spectrogram(audio, self.model.dims.n_mels).to(self.model.device)
        output = self.model.decode(mel, options)
        prediction = output.text.split()

        # Helps a little bit against unstable output and endless repetitions
        if not states.source_finished and len(prediction) > 1:
            prediction = prediction[:-1]

        segment = TextSegment(
            content=" ".join(prediction),
            finished=states.source_finished,
            tgt_lang=self.source_language,
        )

        states.whisper_segments.append(segment)
        
        return WriteAction(segment, finished=states.source_finished)


class NLLBStates(AgentStates):
    def __init__(self):
        super().__init__()
        self.prev_length = 0
        self.previous = []
        self.prefix = []
        self.decoder_ids = []

    def reset(self):
        super().reset()
        self.prev_length = 0
        self.previous = []
        self.prefix = []
        self.decoder_ids = []

    def initialize_decoder_ids(self, ids):
        if len(self.decoder_ids):
            raise Exception("Input not expected, decoder ids already initialized")
        self.decoder_ids = ids


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

    def build_states(self):
        return NLLBStates()
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--source-language", default=LANG_EN, type=str, choices=ALLOWED_LANGUAGES)
        parser.add_argument("--target-language", default=LANG_ET, type=str, choices=ALLOWED_LANGUAGES)

    def policy(self, states: Optional[NLLBStates] = None):
        if states is None:
            states = self.states
        if len(states.source) == 0:
            return ReadAction()
        
        translation_source = states.source
        # Remove empty strings here
        translation_source = list(filter(lambda x: x != "", translation_source))
        if len(translation_source) == 0:
            if states.source_finished:
                return WriteAction("", finished=states.source_finished)
            return ReadAction()
        
        # Nothing new to translate
        if len(translation_source) == states.prev_length and not states.source_finished:
            return ReadAction()
        
        states.prev_length = len(translation_source)

        if len(states.decoder_ids) == 0:
            tgt_lang_id = self.nllb_tokenizer.convert_tokens_to_ids(LANG_TO_NLLB_TOKEN[self.target_language])
            # 2 - </s> token is required for generating a new sentence
            initial_ids = torch.tensor([[2, tgt_lang_id]])
            states.initialize_decoder_ids(initial_ids)

        # Previously transcribed text
        transcribed_text = " ".join(states.source)
        translation_inputs = self.nllb_tokenizer(text=transcribed_text, return_tensors="pt").to(self.device)
        translation_decoder_ids = states.decoder_ids.to(self.device)

        tokens_generated = 0
        while True:
            with torch.no_grad():
                output = self.nllb_model(input_ids=translation_inputs["input_ids"], decoder_input_ids=translation_decoder_ids)

            # Greedy decoding is used
            next_token_logits = output.logits[:, -1, :]
            next_token_scores = next_token_logits.softmax(dim=1)

            next_token = next_token_scores.argmax().unsqueeze(0).unsqueeze(0)

            translation_decoder_ids = torch.cat((translation_decoder_ids, next_token), dim=1).to(self.device)

            if (translation_decoder_ids.squeeze()[-1] == self.nllb_tokenizer.eos_token_id):
                break

            tokens_generated += 1

            if tokens_generated >= 30:
                if translation_decoder_ids.squeeze()[-1] != self.nllb_tokenizer.eos_token_id:
                    eos_token = torch.tensor([[self.nllb_tokenizer.eos_token_id]]).to(self.device)
                    translation_decoder_ids = torch.cat((translation_decoder_ids, eos_token), dim=1).to(self.device)
                break
        
        # If the source is not finished, remove the last token, which usually is 2
        # Even if it is not 2, it still helps with translation quality
        if not states.source_finished and len(translation_decoder_ids[0]) > 2:
            translation_decoder_ids = translation_decoder_ids[..., :-1]

        prediction = self.nllb_tokenizer.decode(translation_decoder_ids[0], skip_special_tokens=True)
        prediction = prediction.split()[len(states.prefix):]

        # Doing Local Agreement prefix matching
        if not states.source_finished:
            confirmed = [] # Words that will be added to the output
            hypothesis = states.prefix + prediction # If all words would be added to the output
            for i in range(len(states.prefix), min(len(states.previous), len(hypothesis))):
                if states.previous[i] == hypothesis[i]:
                    confirmed.append(states.previous[i])
                else:
                    break
        else:
            confirmed = prediction

        states.previous = states.prefix + prediction
        states.prefix = states.prefix + confirmed

        # Update states decoder ids
        if len(confirmed) > 0:
            self.nllb_tokenizer.src_lang = LANG_TO_NLLB_TOKEN[self.target_language]
            tokenized = self.nllb_tokenizer(text=" ".join(states.prefix), return_tensors="pt")["input_ids"]
            # The last token is usually end of sentences
            tokenized = tokenized[:,:-1]
            tokenized = torch.cat((torch.tensor([[2]]), tokenized), dim=1) # Put BOS token back
            states.decoder_ids = tokenized
            self.nllb_tokenizer.src_lang = LANG_TO_NLLB_TOKEN[self.source_language]

        return WriteAction(" ".join(confirmed), finished=states.source_finished)

@entrypoint
class CascadedPipeline(AgentPipeline):
    pipeline = [
        SileroVADAgent,
        WhisperAgent,
        NLLBAgent,
    ]
