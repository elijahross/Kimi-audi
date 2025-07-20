from torch.utils.data import Dataset
from functools import lru_cache
import torch
from typing import Dict, List
from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from kimia_infer.utils.data import KimiAContent
import librosa


def tokenize_message(
    self,
    message,
    tokenize_role=True,
    has_ct_token=False,
    has_msg_end_token=False,
    extract_whisper_feature=False,
    output_type: str = "text",
):
    kimia_content_msg = KimiAContent()

    role = message["role"]
    has_loss = role == "assistant"

    if tokenize_role:
        if role == "user":
            kimia_content_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
        elif role == "assistant":
            kimia_content_msg.audio_append(self.extra_tokens.kimia_assistant_msg_start)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
        else:
            raise NotImplementedError(f"Unknown role: {role}")

    content_blocks = message.get("content", [])
    for block in content_blocks:
        block_type = block.get("type")

        if block_type == "text" or block_type == "phoneme":
            text = block["text"]
            text_tokens = self._tokenize_text(text)

            kimia_content_msg.text_extend(text_tokens, has_loss)
            kimia_content_msg.audio_extend(
                [self.extra_tokens.kimia_text_blank] * len(text_tokens)
            )

            if role == "assistant":
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_eos, has_loss)
                kimia_content_msg.audio_append(self.extra_tokens.kimia_text_blank, audio_token_loss_mask=False)

        elif block_type == "audio":
            audio_tokens = block["content"]["audio_tokens"]  # expected to be pre-tokenized

            kimia_content_msg.audio_append(self.extra_tokens.media_begin)
            kimia_content_msg.audio_extend(audio_tokens, is_continuous=True, audio_token_loss_mask=has_loss)
            kimia_content_msg.audio_append(self.extra_tokens.media_end, audio_token_loss_mask=has_loss)

            kimia_content_msg.text_extend(
                [self.extra_tokens.kimia_text_blank] * (len(audio_tokens) + 2)
            )

            if has_ct_token:
                if output_type == "text":
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
                else:
                    kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ctd_id)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

            if extract_whisper_feature:
                whisper_feature = self.extract_whisper_feat(block["content"]["path"])
                kimia_content_msg.continuous_feature.append(whisper_feature)

        else:
            raise NotImplementedError(f"Unsupported block type: {block_type}")

    if has_msg_end_token:
        kimia_content_msg.audio_append(self.extra_tokens.msg_end, audio_token_loss_mask=False)
        kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

    assert kimia_content_msg.is_valid(), f"Invalid message: {kimia_content_msg}"
    return kimia_content_msg
