---
layout: page
title: TinyWord-134k Model Card
description:
img: assets/img/12.jpg
importance: 1
category: project
related_publications: true
---

> **NOTE:** TinyWord does not use weight-tying, meaning its input and output embedding matrices are separate and untied. At this scale, that roughly doubles the parameter count dedicated to the vocabulary, making the model's performance less impressive than it appears. Furthermore, we plan to train a second version with weight-tying and a new architecture (Qwen3).

## Tiny-Word

Tiny-Word is an extremely tiny Mistral-like model, approximately ~134k parameters. It generates English or Spanish words or word-like sequences.

## Architecture

| Key | Value |
| :---: | :---: |
| hidden_size | 32 |
| num_layers | 2 |
| num_heads | 1 |
| num_kv_heads | 1 |
| intermediate_size | 256 |
| vocab_size | 1200 |
{:.table .table-bordered .table-sm}

## Training

Tiny-Word was trained on 753,232 unique words (entries), 3,225,398 tokens, and 7,022,310 characters. ~660k of those words are English, while ~90k of them are Spanish.

### Dataset

| Key | Value |
| :---: | :---: |
| Entries (words) | 753,232 |
| Tokens | 3,225,398 |
| Characters | 7,022,310 |
| Avg. Tokens Per Entry | ~4.2 |
| Avg. Words Per Entry | 1 |
| Avg. Chars Per Entry | ~9.3 |
| Longest Entry (Tokens) | 36 |
| Shortest Entry (Tokens) | 1 |
| English Words | ~660k |
| Spanish Words | ~90k |
{:.table .table-bordered .table-sm}

### Training Setup

We trained the model for 6 epochs with a batch size of 128 and a gradient accumulation of 2.
The chosen sliding_window was 64, even though the longest word is only 36 tokens, which is inefficient and suboptimal. However, this shouldn't affect the model in any way; it only slows training down.

#### Hardware

Tiny-Word was trained on Google Colaboratory, with 1 Nvidia Tesla T4 GPU, 15 GB of VRAM, and 12.7 GB of RAM.

### Training Results

| step | train_loss | val_loss | train_ppl | val_ppl |
| :--- | :--- | :--- | :--- | :--- |
| 1000 | 4.9619 | 4.5201 | ~143.0 | ~91.8 |
| 3000 | 4.0093 | 3.9156 | ~55.0 | ~50.2 |
| 4000 | 3.8464 | 3.7951 | ~46.8 | ~44.5 |
| 6000 | 3.6814 | 3.6612 | ~39.7 | ~38.9 |
| 7000 | 3.6329 | 3.6182 | ~37.8 | ~37.2 |
| 9000 | 3.5684 | 3.5636 | ~35.5 | ~35.3 |
| 10000 | 3.5452 | 3.5444 | ~34.7 | ~34.6 |
| 12000 | 3.5139 | 3.5161 | ~33.6 | ~33.7 |
| 15000 | 3.4784 | 3.4861 | ~32.4 | ~32.6 |
{:.table .table-bordered .table-sm}

Tiny-Word shows promising results, even at its tiny size (~134k parameters). Given the relatively easy task (predicting subwords inside single words), this is expected.

## Generation Examples

Prompt: `d`
```
desmounder's's's
```

Prompt: `0333333333`
```
ruperperse'sf
```

Prompt: `a`
```
utomatographic'sphon
```

Prompt: `e`
```
equip's's's
```

The model generates plausible word-like sequences that can be pronounced; sometimes it produces real words as well. It can handle almost all input; even if it's nonsensical, it'll still try to generate a word.

## Limitations

1. It does not generate sentences, prose, code, or anything besides a single word-like sequence.
2. It cannot reason or produce complex language.
3. It often appends common artifacts after the word is generated, such as: `'s`, `'sphon`, etc.
4. Most generated words aren't real and instead reflect the lexicon and morphology of the English and Spanish languages.

## Quick Demo

```python
#!/usr/bin/env python3
"""
Tiny Mistral REPL demo — streaming tokens (TextStreamer if available, else manual sampling).
Commands: :quit, :help, :show, :set <param> <value> (max_new_tokens, temperature, top_p, full_output)
"""
from __future__ import annotations
import shlex
import time
import torch
from typing import Optional

from transformers import AutoTokenizer, MistralForCausalLM

# --------- CONFIG ----------
MODEL_DIR = "Harley-ml/TinyWord-134k"
TOKENIZER_DIR = MODEL_DIR
DEFAULT_MAX_NEW_TOKENS = 8 # I don't reccomend going higher than this
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPT = ">>> "
# ---------------------------

def load_tokenizer(path: str):
    print("Loading tokenizer...", path)
    tok = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=False)
    if tok.pad_token is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.add_special_tokens({"pad_token": tok.eos_token})
        else:
            tok.add_special_tokens({"pad_token": "<pad>", "eos_token": "</s>"})
    print("Tokenizer ready. vocab_size=", getattr(tok, "vocab_size", "N/A"))
    return tok

def load_model(path: str, device: str):
    print("Loading model...", path)
    model = None
    try:
        desired_dtype = torch.float16 if device.startswith("cuda") else torch.float32
        model = MistralForCausalLM.from_pretrained(path, local_files_only=False, dtype=desired_dtype)
        print("Loaded with dtype arg.")
    except TypeError:
        model = MistralForCausalLM.from_pretrained(path, local_files_only=False)
        print("Loaded without dtype; will convert.")
    except Exception as e:
        print("Load warning, retrying without dtype:", e)
        model = MistralForCausalLM.from_pretrained(path, local_files_only=False)

    try:
        model.to(device)
        if device.startswith("cuda") and next(model.parameters()).dtype != torch.float16:
            model.half()
        if not device.startswith("cuda") and next(model.parameters()).dtype != torch.float32:
            model.to(torch.float32)
    except Exception as e:
        print("Model move/convert warning:", e)

    model.config.pad_token_id = getattr(model.config, "pad_token_id", None)
    model.eval()
    return model

def top_p_filtering(logits: torch.Tensor, top_p: float, min_keep: int = 1) -> torch.Tensor:
    if top_p <= 0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumprobs = torch.cumsum(probs, dim=-1)
    cutoff = (cumprobs > top_p).nonzero(as_tuple=False)
    if cutoff.numel() > 0:
        idx = int(cutoff[0].item())
        cutoff_idx = max(idx + 1, min_keep)
    else:
        cutoff_idx = sorted_logits.size(-1)
    mask = torch.ones_like(sorted_logits, dtype=torch.bool)
    mask[cutoff_idx:] = False
    filtered = sorted_logits.masked_fill(~mask, -float("inf"))
    return torch.empty_like(filtered).scatter_(0, sorted_idx, filtered)

def manual_stream_generate(model, tokenizer, prompt: str, device: str,
                           max_new_tokens: int = 64, temperature: float = 1.0, top_p: float = 0.9,
                           eos_token_id: Optional[int] = None):
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    past = None
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = getattr(out, "past_key_values", None)

    next_input = input_ids[:, -1:].to(device) if past is not None else input_ids.to(device)
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(input_ids=next_input, past_key_values=past, use_cache=True)
            logits = out.logits[:, -1, :]
            past = getattr(out, "past_key_values", past)

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            filtered = top_p_filtering(logits[0].cpu(), top_p).to(device)
            probs = torch.nn.functional.softmax(filtered.unsqueeze(0), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token[0, 0].item())

        token_text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        yield token_id, token_text

        if eos_token_id is not None and token_id == eos_token_id:
            break
        next_input = torch.tensor([[token_id]], dtype=torch.long, device=device)

def has_text_streamer():
    try:
        from transformers import TextStreamer
        return True
    except Exception:
        return False

class State:
    def __init__(self):
        self.max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        self.temperature = DEFAULT_TEMPERATURE
        self.top_p = DEFAULT_TOP_P
        self.full_output = False
        self.stream = True

def handle_generation(model, tokenizer, prompt: str, device: str, state: State):
    eos = getattr(tokenizer, "eos_token_id", None)
    try:
        if has_text_streamer():
            from transformers import TextStreamer
            streamer = TextStreamer(tokenizer, skip_prompt=not state.full_output, skip_special_tokens=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, add_special_tokens=False)
            inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            inputs.pop("token_type_ids", None)
            model.generate(**inputs,
                           max_new_tokens=state.max_new_tokens,
                           do_sample=True,
                           temperature=state.temperature,
                           top_p=state.top_p,
                           pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id,
                           streamer=streamer)
            print("")
            return
        gen = manual_stream_generate(model, tokenizer, prompt, device,
                                     max_new_tokens=state.max_new_tokens,
                                     temperature=state.temperature,
                                     top_p=state.top_p,
                                     eos_token_id=eos)
        print("GENERATING:", end=" ", flush=True)
        count = 0
        t0 = time.time()
        for _tok_id, tok_text in gen:
            count += 1
            print(tok_text, end="", flush=True)
        print()
        print(f"(generated {count} tokens in {time.time()-t0:.2f}s)")
    except KeyboardInterrupt:
        print("\n[interrupted] Generation aborted by user.")
    except Exception as e:
        print("Generation error:", e)

def repl(model, tokenizer, device):
    state = State()
    help_text = (
        "Commands:\n"
        " :quit\n"
        " :help\n"
        " :show\n"
        " :set <param> <value>  # params: max_new_tokens, temperature, top_p, full_output, stream\n"
        " (blank line repeats last prompt)\n"
    )
    print("Tiny Mistral REPL — device:", device)
    print(help_text)
    last = ""
    while True:
        try:
            raw = input(PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not raw:
            raw = last
            if not raw:
                continue

        if raw.startswith(":"):
            toks = shlex.split(raw)
            cmd = toks[0].lower()
            if cmd == ":quit":
                print("bye.")
                break
            if cmd == ":help":
                print(help_text); continue
            if cmd == ":show":
                print(f"max_new_tokens={state.max_new_tokens}, temperature={state.temperature}, top_p={state.top_p}, full_output={state.full_output}, stream={state.stream}")
                continue
            if cmd == ":set":
                if len(toks) < 3:
                    print("usage: :set <param> <value>"); continue
                k, v = toks[1], toks[2]
                try:
                    if k == "max_new_tokens":
                        state.max_new_tokens = int(v)
                    elif k == "temperature":
                        state.temperature = float(v)
                    elif k == "top_p":
                        state.top_p = float(v)
                    elif k in ("full_output", "full"):
                        state.full_output = v.lower() in ("1", "true", "yes", "y")
                    elif k == "stream":
                        state.stream = v.lower() in ("1", "true", "yes", "y")
                    else:
                        print("unknown param:", k)
                        continue
                    print("OK.")
                except Exception as e:
                    print("set error:", e)
                continue
            print("unknown command")
            continue

        last = raw
        if state.stream:
            handle_generation(model, tokenizer, raw, device, state)
        else:
            try:
                inputs = tokenizer(raw, return_tensors="pt", truncation=True, add_special_tokens=False)
                inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
                inputs.pop("token_type_ids", None)
                out = model.generate(**inputs,
                                     max_new_tokens=state.max_new_tokens,
                                     do_sample=True,
                                     temperature=state.temperature,
                                     top_p=state.top_p,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
                seq = out[0]
                input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
                text = tokenizer.decode(seq if state.full_output else seq[input_len:], skip_special_tokens=True)
                print("\nOUTPUT\n", text)
            except Exception as e:
                print("Generation failed:", e)

def main():
    device = DEVICE
    tokenizer = load_tokenizer(TOKENIZER_DIR)
    model = load_model(MODEL_DIR, device)
    repl(model, tokenizer, device)

if __name__ == "__main__":
    main()
```

## Related Models

1. [PicoWord](https://huggingface.co/Harley-ml/PicoWord-5k)
2. [MicroWord](https://huggingface.co/Harley-ml/MicroWord-23k)
3. [TinyWord2](https://huggingface.co/Harley-ml/TinyWord2-128k)
4. [MediumWord](https://huggingface.co/Harley-ml/MediumWord-559k)
