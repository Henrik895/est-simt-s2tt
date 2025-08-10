from transformers.utils import logging

logging.set_verbosity_error()

from simuleval.agents import AgentPipeline, SpeechToTextAgent
from simuleval.agents import ReadAction, WriteAction
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from typing import Optional
from examples.demo.silero_vad import SileroVADAgent
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

import numpy as np
import torch
from pathlib import Path

import re


# The problem is that seamless fine-tuning code and HuggingFace use
# different layer names, so the layer names have to be mapped back.
# Might not be the most efficient sequence of regex, but all the
# layers get correct names in the end.
# Initially 1431 mismatches.
def map_unexpected_to_missing(unexpected_name: str) -> str:
    name = unexpected_name.replace("module.model.", "")
    name = name.replace("speech_encoder_frontend.post_extract_layer_norm", "speech_encoder.feature_projection.layer_norm")
    name = name.replace("speech_encoder_frontend.model_dim_proj", "speech_encoder.feature_projection.projection")
    name = name.replace("speech_encoder.inner", "speech_encoder.encoder")
    name = name.replace(".ffn1.inner_proj", ".ffn1.intermediate_dense")
    name = name.replace(".ffn1.output_proj", ".ffn1.output_dense")
    name = name.replace(".ffn2.inner_proj", ".ffn2.intermediate_dense")
    name = name.replace(".ffn2.output_proj", ".ffn2.output_dense")
    name = name.replace(".self_attn.q_proj", ".self_attn.linear_q")
    name = name.replace(".self_attn.k_proj", ".self_attn.linear_k")
    name = name.replace(".self_attn.v_proj", ".self_attn.linear_v")
    name = name.replace(".self_attn.output_proj", ".self_attn.linear_out")
    name = name.replace(".self_attn.sdpa.rel_k_embed", ".self_attn.distance_embedding")
    name = name.replace(".conv_layer_norm", ".conv_module.layer_norm")
    name = name.replace(".conv.pointwise_conv1", ".conv_module.pointwise_conv1")
    name = name.replace(".conv.depthwise_conv", ".conv_module.depthwise_conv")
    name = name.replace(".conv.layer_norm", ".conv_module.depthwise_layer_norm")
    name = name.replace(".conv.pointwise_conv2", ".conv_module.pointwise_conv2")
    name = name.replace(".layer_norm", ".final_layer_norm")
    name = name.replace(".conv_module.final_layer_norm.", ".conv_module.layer_norm.")
    name = name.replace("speech_encoder.adaptor_layers.", "speech_encoder.adapter.layers.")
    name = name.replace(".encoder_decoder_attn.", ".cross_attention.")
    name = name.replace(".cross_attention.output_proj.", ".cross_attention.out_proj.")
    name = name.replace(".inner_proj.", ".fc1.")
    name = name.replace(".output_proj.", ".fc2.")
    name = name.replace(".encoder_decoder_attn_layer_norm.", ".cross_attention_layer_norm.")
    name = name.replace(".self_attn.linear_out.", ".self_attn.out_proj.")
    name = re.sub(r'(text_decoder\.layers\.\d+\.self_attn\.)linear_q\b', r'\1q_proj', name)
    name = re.sub(r'(text_decoder\.layers\.\d+\.self_attn\.)linear_k\b', r'\1k_proj', name)
    name = re.sub(r'(text_decoder\.layers\.\d+\.self_attn\.)linear_v\b', r'\1v_proj', name)
    name = re.sub(r'(speech_encoder\.encoder\.layers\.\d+\.self_attn\.)out_proj\b', r'\1linear_out', name)
    name = name.replace("speech_encoder.adapter.layers.0.self_attn.out_proj.", "speech_encoder.adapter.layers.0.self_attn.linear_out.")
    name = name.replace("speech_encoder.adapter.layers.0.ffn.fc1.", "speech_encoder.adapter.layers.0.ffn.intermediate_dense.")
    name = name.replace("speech_encoder.adapter.layers.0.ffn.fc2.", "speech_encoder.adapter.layers.0.ffn.output_dense.")
    name = name.replace("speech_encoder.proj1.", "speech_encoder.intermediate_ffn.intermediate_dense.")
    name = name.replace("speech_encoder.proj2.", "speech_encoder.intermediate_ffn.output_dense.")
    name = name.replace("final_proj.weight", "lm_head.weight")
    name = name.replace("text_decoder.final_layer_norm.", "text_decoder.layer_norm.")
    name = name.replace("text_decoder.final_layer_norm.", "text_decoder.layer_norm.")
    name = name.replace("speech_encoder.feature_projection.final_layer_norm.", "speech_encoder.feature_projection.layer_norm.")
    name = name.replace("speech_encoder.encoder_layer_norm.", "speech_encoder.encoder.layer_norm.")
    name = name.replace("speech_encoder.final_layer_norm.", "speech_encoder.inner_layer_norm.")
    name = name.replace("text_decoder_frontend.embed.weight", "text_decoder.embed_tokens.weight")
    name = name.replace("text_encoder_frontend.embed.weight", "shared.weight")

    return name


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

SEAMLESS_BENCHMARK = "facebook/seamless-m4t-v2-large" # The model loaded from the HuggingFace
SEAMLESS_BENCHMARK_LOCAL = Path("checkpoint.pt") # Fine-tuned weights, that will be loaded onto the other model
SAMPLING_RATE = 16000

class SeamlessStates(AgentStates):
    def __init__(self):
        super().__init__()
        self.previous = []
        self.prefix = []

    def reset(self):
        super().reset()
        self.previous = []
        self.prefix = []
    

class SeamlessAgent(SpeechToTextAgent):
    def __init__(self, args = None):
        super().__init__(args)
        self.wait_k = args.wait_k
        self.source_language = args.source_language
        self.target_language = args.target_language
        self.device = args.device
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.dtype = "float16" if torch.cuda.is_available() else "float32"
        self.processor = AutoProcessor.from_pretrained(SEAMLESS_BENCHMARK)
        # Load fine-tuned weights into the model
        checkpoint = torch.load(SEAMLESS_BENCHMARK_LOCAL, map_location="cpu")
        checkpoint_state = checkpoint['model']
        hf_state = {}
        for k, v in checkpoint_state.items():
            mapped_k = map_unexpected_to_missing(k)
            hf_state[mapped_k] = v

        self.model = SeamlessM4Tv2ForSpeechToText.from_pretrained(SEAMLESS_BENCHMARK, torch_dtype=self.torch_dtype).to(self.device)
        missing, unexpected = self.model.load_state_dict(state_dict=hf_state, strict=True)
        # missing and unexpected can be used for debugging, but then set strict=False,
        # otherwise error will be thrown and the layer names are not written into the
        # missing and unexpect files

        # with open("missing.txt", "w+") as f:
        #     for name in missing:
        #         f.write(f"{name}\n")
        # with open("unexpected.txt", "w+") as f:
        #     for name in unexpected:
        #         f.write(f"{name}\n")
        print("Seamless agent")
        print("seamless:", SEAMLESS_BENCHMARK)
        print("Device:", self.device)
        print("Torch dtype:", self.torch_dtype)
        print("wait-k:", self.wait_k)
        print("source (not used):", self.source_language)
        print("target:", self.target_language)

    def build_states(self):
        return SeamlessStates()
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--wait-k", default=2, type=int)
        parser.add_argument("--source-language", default=LANG_EST, choices=LANG_CHOICES)
        parser.add_argument("--target-language", default=LANG_ENG, choices=LANG_CHOICES)

    def policy(self, states: Optional[SeamlessStates] = None):
        # Input from VADAgent
        # SpeechSegment(index=0, content=..., finished=False, is_empty=False, data_type='speech', tgt_lang=None, config={}, sample_rate=16000)
        if states is None:
            states = self.states
        if states.source_sample_rate == 0:
            length_in_seconds = 0
        else:
            length_in_seconds = round(len(states.source) / states.source_sample_rate)

        if not states.source_finished and length_in_seconds < self.wait_k:
            return ReadAction()
        
        if len(states.source) == 0:
            if states.source_finished:
                return WriteAction("", finished=states.source_finished)
            return ReadAction()

        prefix = " ".join(states.prefix)

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
        if len(prefix):
            prefix_ids = self.processor(text=prefix, return_tensors="pt")["input_ids"].to(self.device)
            # Remove special tokens
            if prefix_ids[0][-1] == 0:
                prefix_ids = prefix_ids[:, :-1]
            if prefix_ids[0][-1] == 3:
                prefix_ids = prefix_ids[:, :-1]
            prefix_ids = prefix_ids[:,1:] # Add target language token to the beginning
            language_prefix = torch.tensor([[3, LANG_TO_TOKEN[self.target_language]]]).to(self.device)
            prefix_ids = torch.cat((language_prefix, prefix_ids), dim=1).to(self.device)
        else:
            prefix_ids = torch.tensor([[3, LANG_TO_TOKEN[self.target_language]]]).to(self.device)
        
        source = np.array(states.source).astype(self.dtype)
        audio_inputs = self.processor(audios=source, sampling_rate=states.source_sample_rate, return_tensors="pt").to(self.device)
        text_tokens = self.model.generate(**audio_inputs, decoder_input_ids=prefix_ids, do_sample=False, num_beams=5, max_new_tokens=30)
        prediction = self.processor.decode(text_tokens[0], skip_special_tokens=True).split()[len(states.prefix):]

        # Local Agreement prefix matchines
        if not states.source_finished:
            confirmed = [] # Output words
            hypothesis = states.prefix + prediction # All possible words
            for i in range(len(states.prefix), min(len(states.previous), len(hypothesis))):
                if states.previous[i] == hypothesis[i]:
                    confirmed.append(states.previous[i])
                else:
                    break
        else:
            confirmed = prediction

        states.previous = states.prefix + prediction
        states.prefix = states.prefix + confirmed
        
        return WriteAction(" ".join(confirmed), finished=states.source_finished)

@entrypoint
class ExperimentalPipeline(AgentPipeline):
    pipeline = [
        SileroVADAgent,
        SeamlessAgent,
    ]
