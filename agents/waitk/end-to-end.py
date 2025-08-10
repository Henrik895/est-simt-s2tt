from transformers.utils import logging

logging.set_verbosity_error()

from simuleval.agents import AgentPipeline, SpeechToTextAgent
from simuleval.agents import ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText
from typing import Optional

import torch
import numpy as np


LANG_EST = "est"
LANG_ENG = "eng"
LANG_RUS = "rus"

LANG_CHOICES = [
    LANG_EST,
    LANG_ENG,
    LANG_RUS,
]

TOKEN_EST = 256023
TOKEN_ENG = 256022
TOKEN_RUS = 256074

LANG_TO_TOKEN = {
    LANG_EST: TOKEN_EST,
    LANG_ENG: TOKEN_ENG,
    LANG_RUS: TOKEN_RUS,
}

SEAMLESS_BENCHMARK = "facebook/seamless-m4t-v2-large"
SAMPLING_RATE = 16000


class SeamlessStates(AgentStates):
    def __init__(self):
        super().__init__()
        self.previous = []

    def reset(self):
        super().reset()
        self.previous = []

    def initialize_decoder_ids(self, ids):
        if len(self.decoder_ids):
            raise Exception("Input not expected, decoder ids already initialized")
        self.decoder_ids = ids


class SeamlessAgent(SpeechToTextAgent):
    def __init__(self, args = None):
        super().__init__(args)
        self.wait_k = args.wait_k
        self.source_language = args.source_language
        self.target_language = args.target_language
        self.processor = AutoProcessor.from_pretrained(SEAMLESS_BENCHMARK)
        self.device = args.device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.dtype = "float16" if torch.cuda.is_available() else "float32"
        print("Seamless agent")
        print("Device:", self.device)
        print("Torch dtype:", self.torch_dtype)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(SEAMLESS_BENCHMARK, torch_dtype=self.torch_dtype).to(self.device)

    def build_states(self):
        return SeamlessStates()

    @staticmethod
    def add_args(parser):
        parser.add_argument("--wait-k", default=2, type=int)
        parser.add_argument("--source-language", default=LANG_EST, choices=LANG_CHOICES)
        parser.add_argument("--target-language", default=LANG_ENG, choices=LANG_CHOICES)

    def policy(self, states: Optional[SeamlessStates] = None):
        if states is None:
            states = self.states
        if states.source_sample_rate == 0:
            length_in_seconds = 0
        else:
            length_in_seconds = round(len(states.source) / states.source_sample_rate)

        if not states.source_finished and length_in_seconds < self.wait_k:
            return ReadAction()
        
        prefix_ids = None
        if len(states.previous) == 0:
            """
            For continuing generation.
            Special tokens IDs:
                *      3 - </s>
                * 256022 - __eng__
                * 256023 - __est__
                * 256074 - __rus__

            If using forced_decoder_ids, the sequence should start with 3 and then a language token.
            Do not pass tgt_lang attribute to model.generate because it overwrites decoder_input_ids content.
            """
            prefix_ids = torch.tensor([[3, LANG_TO_TOKEN[self.target_language]]]).to(self.device)
        else:
            prefix = " ".join(states.previous) # Generate tokens based on previous output
            prefix_ids = self.processor(text=prefix, return_tensors="pt")["input_ids"].to(self.device)
            # Check for special tokens
            if prefix_ids[0][-1] == 0:
                prefix_ids = prefix_ids[:, :-1]
            if prefix_ids[0][-1] == 3:
                prefix_ids = prefix_ids[:, :-1]
            prefix_ids = prefix_ids[:,1:] # We will add our own prefix
            language_prefix = torch.tensor([[3, LANG_TO_TOKEN[self.target_language]]]).to(self.device)
            prefix_ids = torch.cat((language_prefix, prefix_ids), dim=1).to(self.device)

        source = np.array(states.source).astype(self.dtype)
        audio_inputs = self.processor(audios=source, sampling_rate=states.source_sample_rate, return_tensors="pt").to(self.device)
        text_tokens = self.model.generate(**audio_inputs, decoder_input_ids=prefix_ids, do_sample=False, num_beams=5, max_new_tokens=30)

        # Remove last token
        if not states.source_finished and len(text_tokens[0]) > 2:
            text_tokens = text_tokens[..., :-1]

        prediction = self.processor.decode(text_tokens[0], skip_special_tokens=True).split()[len(states.previous):]

        # Last word may not be finished, so remove it
        if not states.source_finished and len(prediction) > 1:
            prediction = prediction[:-1]

        states.previous = states.previous + prediction

        return WriteAction(content=" ".join(prediction), finished=states.source_finished)


@entrypoint
class EndToEndPipeline(AgentPipeline):
    pipeline = [SeamlessAgent]
