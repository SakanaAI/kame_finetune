from typing import Any

import torch
from kame.models import LMModel
from kame.utils.compile import CUDAGraphed
from kame.utils.sampling import sample_token

from models.moshi_for_finetuning import MoshiForFinetuning


class MoshiForConditionalGeneration:
    def __init__(self, moshi_lm: LMModel | MoshiForFinetuning):
        self.moshi_lm = moshi_lm

    def prepare_generation(
        self,
        batch_size: int,
        text_sampling_params: dict[str, Any],
        audio_sampling_params: dict[str, Any],
    ) -> None:
        self.text_sampling_params = text_sampling_params
        self.audio_sampling_params = audio_sampling_params
        self.tempformer_forward_graphed = CUDAGraphed(self.moshi_lm.forward_text)
        self.depformer_step_graphed = CUDAGraphed(self.depformer_step)
        self.moshi_lm.streaming_forever(batch_size)

    def finish_generation(self):
        self.text_sampling_params = None
        self.audio_sampling_params = None
        self.tempformer_forward_graphed = None
        self.depformer_step_graphed = None
        self.moshi_lm.reset_streaming()

    def depformer_step(
        self, text_token: torch.LongTensor, transformer_out: torch.Tensor
    ) -> torch.LongTensor:
        """
        Generate next audio tokens from the given text token.

        Args:
            text_token (torch.Tensor):
                The text token to generate from. shape is [B]
            transformer_out (torch.Tensor):
                The output of the transformer. shape is [B, 1, dim]

        Returns:
            audio_tokens (torch.Tensor):
                The generated audio tokens. shape is [B, K]
        """
        (B,) = text_token.shape
        prev_token = text_token
        depformer_tokens: list[torch.Tensor] = []
        assert not self.moshi_lm.depformer.is_streaming
        with self.moshi_lm.depformer.streaming(B):
            for cb_index in range(self.moshi_lm.dep_q):
                input_ = prev_token[:, None, None]
                logits = self.moshi_lm.forward_depformer(cb_index, input_, transformer_out)
                next_token = sample_token(logits.float(), **self.audio_sampling_params)
                assert next_token.shape == (B, 1, 1)
                next_token = next_token[:, 0, 0]  # shape is B
                depformer_tokens.append(next_token)
                prev_token = next_token

        assert len(depformer_tokens) == self.moshi_lm.dep_q, (
            len(depformer_tokens),
            self.moshi_lm.dep_q,
        )
        out = torch.stack(depformer_tokens, dim=1)
        assert out.shape == (B, self.moshi_lm.dep_q), out.shape
        return out

    @torch.no_grad()
    def step(self, last_tokens: torch.LongTensor) -> torch.LongTensor:
        """
        Generate next tokens from the given last tokens.

        Args:
            last_tokens (torch.LongTensor):
                The last tokens to generate from.
                shape is [B, K, T], where B is the batch size, K is the number of codebooks, and T is the number of tokens.
                T > 1 means it is the first step of generation when prompt tokens are given.

        Returns:
            sampled_tokens (torch.LongTensor):
                The sampled tokens. shape is [B, K, 1]
        """

        if not self.moshi_lm.transformer.is_streaming:
            raise RuntimeError("moshi_lm.transformer should be in streaming mode")

        B, K, T = last_tokens.shape
        assert K == self.moshi_lm.num_codebooks, (
            f"Expected {self.moshi_lm.num_codebooks}, but got {K}"
        )

        # 1 tempral transformer forward
        if T > 1:  # cannot forward multiple tokens to the graphed function
            temp_out, text_logits = self.moshi_lm.forward_text(last_tokens)
        else:
            temp_out, text_logits = self.tempformer_forward_graphed(last_tokens)
        assert temp_out.shape == (
            B,
            T,
            self.moshi_lm.dim,
        ), f"Expected shape {(B, T, self.moshi_lm.dim)}, but got {temp_out.shape}"
        assert text_logits.shape == (
            B,
            1,
            T,
            self.moshi_lm.text_card,
        ), f"Expected shape {(B, 1, T, self.moshi_lm.text_card)}, but got {text_logits.shape}"

        temp_out = temp_out[:, -1][:, None]  # shape is [B, 1, dim]
        text_logits = text_logits[:, :, -1]  # shape is [B, 1, text_card]

        # 2 sample next text tokens
        text_token = sample_token(text_logits.float(), **self.text_sampling_params)
        assert text_token.shape == (B, 1), f"Expected shape {(B, 1)}, but got {text_token.shape}"
        text_token = text_token[:, 0]  # shape is [B]

        # 3 sample next audio tokens
        audio_tokens = self.depformer_step_graphed(text_token, temp_out)
        assert audio_tokens.shape == (B, self.moshi_lm.dep_q), audio_tokens.shape

        sampled_tokens = torch.cat([text_token[:, None], audio_tokens], dim=-1)
        assert sampled_tokens.shape == (B, self.moshi_lm.num_codebooks), sampled_tokens.shape
        sampled_tokens = sampled_tokens[..., None]  # shape is [B, num_codebooks, 1]

        return sampled_tokens

    def generate(
        self,
        prompt_tokens: torch.LongTensor,
        generation_length: int,
        text_sampling_params: dict[str, Any],
        audio_sampling_params: dict[str, Any],
    ) -> torch.LongTensor:
        """
        Generate text and audio streams from the given prompt tokens.
        Make sure that prompt and output tokens are delayed by moshi_lm.delays.

        Args:
            prompt_tokens (torch.LongTensor):
                The prompt tokens to generate from.
                shape is [B, K, Tp], where B is the batch size, K is the number of codebooks, and Tp is the number of tokens.
            generation_length (int):
                The number of tokens to generate.
            text_sampling_params (dict[str, Any]):
                The sampling parameters for text tokens.
            audio_sampling_params (dict[str, Any]):
                The sampling parameters for audio tokens.

        Returns:
            generated_tokens (torch.LongTensor):
                The generated tokens. shape is [B, K, Tg]
        """

        B, K, Tp = prompt_tokens.shape
        assert K == self.moshi_lm.num_codebooks, (
            f"Expected {self.moshi_lm.num_codebooks}, but got {K}"
        )

        # prepare generation
        self.prepare_generation(B, text_sampling_params, audio_sampling_params)

        # generate
        list_of_tokens = []
        last_tokens = prompt_tokens
        for _ in range(generation_length):
            last_tokens = self.step(last_tokens)
            list_of_tokens.append(last_tokens)

        # end generation
        self.finish_generation()

        generated_tokens = torch.cat(list_of_tokens, dim=-1)
        assert generated_tokens.shape == (B, K, generation_length), generated_tokens.shape

        return generated_tokens
