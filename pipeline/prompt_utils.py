from typing import List, Union

import torch


def encode_edit_prompt(tokenizer, text_encoder, editing_prompt, device):
    if isinstance(editing_prompt, list):
        encoded = []
        for item in editing_prompt:
            if isinstance(item, str):
                input_ids = tokenizer(
                    item,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    return_tensors="pt",
                ).input_ids.to(device)
                encoded.append(text_encoder(input_ids)[0])
            else:
                encoded.append(item)
        return encoded

    if isinstance(editing_prompt, str):
        input_ids = tokenizer(
            editing_prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids.to(device)
        return text_encoder(input_ids)[0]

    return editing_prompt


def encode_empty_prompt(tokenizer, text_encoder, device):
    input_ids = tokenizer(
        "",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    return text_encoder(input_ids)[0]