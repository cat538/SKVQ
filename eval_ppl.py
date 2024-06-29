import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    # LlamaForCausalLM,
    AutoTokenizer,
)
from experiments.modeling_llama_skvq import LlamaForCausalLM
from experiments.utils import plug_quantizer_into_model
from KVcache_manager import ModelKVCacheManager
from calib_config import *


@torch.no_grad()
def eval_ppl(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset="wikitext2-v1",
    input_len: int = 2048,
):
    if dataset not in DATASET_CACHE:
        raise RuntimeError(f"{dataset} invalid")

    testdata = load_dataset(DATASET_CACHE["wikitext2-v1"], split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")["input_ids"]

    nsamples = testenc.numel() // input_len
    nlls = []

    loss_fct = nn.CrossEntropyLoss()
    for i in tqdm(range(nsamples)):
        # [bs, input_len]
        batch = testenc[:, (i * input_len) : ((i + 1) * input_len)].to(model.device)
        outputs = model.model(batch)
        hidden_states = outputs[0]
        # [bs, input_len, vocab_size]
        logits = model.lm_head(hidden_states)
        # [bs, input_len-1, vocab_size]
        shift_logits = logits[:, :-1, :]
        # [bs, input_len-1]
        shift_labels = batch[:, 1:].to(model.lm_head.weight.device)
        loss = loss_fct(
            # [bs * (input_len-1), vocab_size]
            shift_logits.view(-1, shift_logits.size(-1)),
            # [bs * (input_len-1)]
            shift_labels.view(-1),
        )
        neg_log_likelihood = loss.float() * input_len
        nlls.append(neg_log_likelihood)

        for layer in model.model.layers:
            manager = getattr(layer.self_attn, "KV_cache_manager", None)
            if manager is not None:
                manager.clear()

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * input_len)).item()
    print(dataset, ppl)

    return ppl


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)

    model_to_len = {
        "llama-7b": 2048,
        "llama2-7b": 4096,
        "llama2-13b": 4096,
        "llama2-7b-chat": 4096,
        "llama2-13b-chat": 4096,
        "mistral-7b": 8192,
        "llama2-7b-80k": 10000,
    }

    args = parser.parse_args()
    model_name = args.model
    MODEL_PATH = MODEL_NAME_TO_PATH[model_name]

    # 0. load model and tokenizer
    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    num_layers = len(model.model.layers)
    input_len = model_to_len[model_name]

    # 1. fp16 baseline
    fp16_ppl = eval_ppl(model, tokenizer, input_len=input_len)

    # 2. create ModelKVCacheManager
    kv_managers_lis: list[ModelKVCacheManager] = []

    group_set = [64]
    for group_size in group_set:
        rod_meta = MODEL_TO_REORDER[model_name][group_size]["minmax"]
        for kbits, vbits in [
            (4,4), (3,3), (2,2),
        ]:
            kv_managers_lis.append(
                ModelKVCacheManager.create(
                    model,
                    kbits,
                    vbits,
                    group_size,
                    reorder_file=rod_meta,
                    smooth_file=None,
                    window_size=0,
                    pre_rope=True,
                    clipping=[0.96 for _ in range(num_layers)],
                    attn_sink=5,
                    full_prefill=False,
                    fp8=True,
                    fake_quant=True,
                )
            )

    # 3. SKVQ PPL
    for model_kv_manager in kv_managers_lis:
        model_kv_manager.full_prefill(False)
        plug_quantizer_into_model(model, model_kv_manager)
        print(model_kv_manager)
        ppl = eval_ppl(model, tokenizer, input_len=input_len)
