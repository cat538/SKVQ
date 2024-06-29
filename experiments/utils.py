import six
import operator

from collections import Counter
from transformers import PreTrainedModel

from KVcache_manager import ModelKVCacheManager
from experiments.modeling_llama_skvq import LlamaForCausalLM
from experiments.modeling_mistral_skvq import MistralForCausalLM


def cal_f1(precision: float, recall: float):
    if precision + recall > 0:
        fmeasure = 2 * precision * recall / (precision + recall)
    else:
        fmeasure = 0.0
    return fmeasure


def rouge1(ref: list[int], pred: list[int]):
    ref_counter = Counter(ref)
    pred_counter = Counter(pred)
    intersection = 0

    for ngram in six.iterkeys(ref_counter):
        intersection += min(ref_counter[ngram], pred_counter[ngram])

    precision = intersection / max(len(pred), 1)
    recall = intersection / max(len(ref), 1)

    return precision, recall, cal_f1(precision, recall)


def rougeL(ref: list[int], pred: list[int]):
    """
    calculate LCS with dp
    """
    rows = len(ref)
    cols = len(pred)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            if ref[i - 1] == pred[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[-1][-1]

    precision = lcs / max(len(pred), 1)
    recall = lcs / max(len(ref), 1)

    return precision, recall, cal_f1(precision, recall)


def plug_quantizer_into_model(
    model: PreTrainedModel, model_kv_manager: ModelKVCacheManager
):
    if model_kv_manager is None:
        return

    kv_managers = model_kv_manager.kv_managers
    proj_map = {
        LlamaForCausalLM: {
            "k": operator.attrgetter("self_attn.k_proj"),
            "v": operator.attrgetter("self_attn.v_proj"),
        },
        MistralForCausalLM: {
            "k": operator.attrgetter("self_attn.k_proj"),
            "v": operator.attrgetter("self_attn.v_proj"),
        },
    }
    wname_map: dict[str, str] = proj_map.get(model.__class__, None)
    assert wname_map is not None, f"Not supported for {model.__class__}"

    for idx, layer in enumerate(model.model.layers):
        kv_manager = kv_managers[idx]
        layer.self_attn.KV_cache_manager = kv_manager

    model.model.model_kv_manager = model_kv_manager
    model.model_kv_manager = model_kv_manager
