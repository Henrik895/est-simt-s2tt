from transformers.utils import logging

logging.set_verbosity_error()

from simuleval.agents import AgentPipeline, SpeechToTextAgent
from simuleval.agents import ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from typing import Optional
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

import numpy as np
import torch


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


class SeamlessAgent(SpeechToTextAgent):
    def __init__(self, args = None):
        super().__init__(args)
        self.target_language = args.target_language
        self.device = args.device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.dtype = "float16" if torch.cuda.is_available() else "float32"
        self.processor = AutoProcessor.from_pretrained(SEAMLESS_BENCHMARK)
        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(SEAMLESS_BENCHMARK, torch_dtype=self.torch_dtype).to(self.device)
        print("Seamless agent")
        print("Device:", self.device)
        print("Torch dtype:", self.torch_dtype)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--target-language", default=LANG_ENG, choices=LANG_CHOICES)

    def policy(self, states: Optional[AgentStates] = None):
        if states is None:
            states = self.states

        # Translate only at the very end, currenly doing regular translation
        if not states.source_finished:
            return ReadAction()
        
        source = np.array(states.source).astype(self.dtype)
        audio_inputs = self.processor(audios=source, sampling_rate=states.source_sample_rate, return_tensors="pt").to(self.device)
        text_tokens = self.model.generate(**audio_inputs, do_sample=False, num_beams=5, tgt_lang=self.target_language, max_new_tokens=250)
        prediction = self.processor.decode(text_tokens[0], skip_special_tokens=True)
        
        return WriteAction(prediction, finished=states.source_finished)

@entrypoint
class EndToEndPipeline(AgentPipeline):
    pipeline = [
        SeamlessAgent,
    ]
