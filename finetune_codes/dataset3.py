from torch.utils.data import Dataset
from functools import lru_cache
import torch
from typing import Dict, List
from kimia_infer.utils.special_tokens import instantiate_extra_tokens
from kimia_infer.utils.data import KimiAContent


class LazySupervisedDataset(Dataset):
    """Text-only multiturn dataset for supervised fine-tuning."""

    def __init__(self, raw_data_list, text_tokenizer, max_len: int, kimia_token_offset: int):
        super().__init__()
        self.max_len = max_len
        self.text_tokenizer = text_tokenizer
        self.extra_tokens = instantiate_extra_tokens(self.text_tokenizer)

        self.pad_token = self.extra_tokens.pad
        self.kimia_token_offset = kimia_token_offset
        self.raw_data = raw_data_list

        print("There are {} samples in the dataset".format(len(raw_data_list)))

    def __len__(self):
        return len(self.raw_data)

    def _tokenize_text(self, text):
        if text is None:
            return None
        return self.text_tokenizer.encode(text, bos=False, eos=False)

    def tokenize_message(
        self,
        message,
        tokenize_role=True,
        has_ct_token=False,
        has_msg_end_token=False,
    ):
        kimia_content_msg = KimiAContent()
        role = message["role"]
        has_loss = role == "assistant"

        if message["message_type"] != "text":
            raise ValueError(f"Expected only 'text' messages, got: {message['message_type']}")

        if tokenize_role:
            if role == "user":
                kimia_content_msg.audio_append(self.extra_tokens.kimia_user_msg_start)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            elif role == "assistant":
                kimia_content_msg.audio_append(self.extra_tokens.kimia_assistant_msg_start)
                kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)
            else:
                raise NotImplementedError(f"role: {role}")

        text = message["content"]
        text_tokens = self._tokenize_text(text)

        kimia_content_msg.text_extend(text_tokens, has_loss)
        kimia_content_msg.audio_extend([self.extra_tokens.kimia_text_blank] * len(text_tokens))

        if role == "assistant":
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_eos, has_loss)
            kimia_content_msg.audio_append(self.extra_tokens.kimia_text_blank, audio_token_loss_mask=False)

        if has_ct_token:
            kimia_content_msg.audio_append(self.extra_tokens.kimia_speech_ct_id)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

        if has_msg_end_token:
            kimia_content_msg.audio_append(self.extra_tokens.msg_end, audio_token_loss_mask=False)
            kimia_content_msg.text_append(self.extra_tokens.kimia_text_blank)

        assert kimia_content_msg.is_valid(), f"Invalid kimia_content_msg: {kimia_content_msg}"
        return kimia_content_msg

    def tokenize_conversation(
        self, messages: List[Dict], output_type: str = "text", add_assistant_start_msg: bool = True
    ) -> KimiAContent:
        assert output_type == "text"

        msgs: List[KimiAContent] = []
        previous_role = None

        for msg_idx, message in enumerate(messages):
            assert message["role"] in ["user", "assistant"]

            tokenize_role = previous_role is None or message["role"] != previous_role
            previous_role = message["role"]

            if msg_idx == len(messages) - 1:
                has_ct_token = True
                has_msg_end_token = True
            else:
                next_role = messages[msg_idx + 1]["role"]
                has_ct_token = message["role"] != next_role
                has_msg_end_token = has_ct_token

            msg = self.tokenize_message(
                message=message,
                tokenize_role=tokenize_role,
                has_ct_token=has_ct_token,
                has_msg_end_token=has_msg_end_token,
            )
            msgs.append(msg)

        if add_assistant_start_msg:
            assistant_start_msg = self.tokenize_message(
                {
                    "role": "assistant",
                    "message_type": None,
                    "content": None,
                },
                tokenize_role=True,
                has_ct_token=False,
                has_msg_end_token=False,
            )
            msgs.append(assistant_start_msg)

        ret_msg = msgs[0]
        for msg in msgs[1:]:
            ret_msg.merge(msg)
        return ret_msg

    @lru_cache(maxsize=None)
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        conversation = self.raw_data[i]["conversation"]
        tokenized_conversation = self.tokenize_conversation(
            conversation, output_type="text", add_assistant_start_msg=False
        )

        _, text_input_ids, _, _, text_token_loss_mask = tokenized_conversation.to_tensor()

        text_labels = torch.cat(
            (text_input_ids[:, 1:], text_input_ids.new_full((1, 1), self.pad_token)), dim=1
        )
        text_loss_mask = torch.cat(
            (text_token_loss_mask[:, 1:], text_token_loss_mask.new_full((1, 1), False)), dim=1
        )

        return {
            "text_input_ids": text_input_ids,
            "labels": text_labels,
            "text_loss_mask": text_loss_mask,
        }

    @staticmethod
    def collate_fn(batch):
        assert len(batch) == 1, "Micro batch size must be 1"
        return batch[0]
