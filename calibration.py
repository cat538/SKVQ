import os
import torch
import argparse
import random
from tqdm import tqdm
from functools import partial
from datasets import load_dataset
from sklearn.cluster import KMeans
from typing import Literal
from transformers import(
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from calib_config import MODEL_NAME_TO_PATH, DATASET_CACHE


def get_wikitext2(seed, nsamples, seqlen, tokenizer):
    traindata = load_dataset(DATASET_CACHE["wikitext2-v1"], split='train')
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_ptb(seed, nsamples, seqlen, tokenizer):
    traindata = load_dataset(DATASET_CACHE["ptb"], split='train')
    trainenc = tokenizer("\n\n".join(traindata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader


def get_data(
    dataset: str,
    tokenizer: PreTrainedTokenizer,
    sample_len=20000,
    nsample=128,
) -> list[torch.Tensor]:
    if dataset == "needle":
        return torch.load(DATASET_CACHE["needle"])

    seed = 42
    if dataset == "wikitext2-v1":
        data = get_wikitext2(seed, nsample, sample_len, tokenizer)
    elif dataset == "ptb":
        data = get_ptb(seed, nsample, sample_len, tokenizer)
    else:
        raise RuntimeError("Not supported")

    return data


@torch.no_grad()
def calibration(
    model: PreTrainedModel,
    dataset: str,
    tokenizer: PreTrainedTokenizer,
    sample_len:int=20000,
    nsample:int=128,
    num_hist_bin=128,
    smooth_scale: list[torch.Tensor] = None,
):
    model = model.model
    layers = model.layers

    res = {
        "min": {
            "q":  [None for _ in range(len(layers))],
            "k":  [None for _ in range(len(layers))],
            "v":  [None for _ in range(len(layers))],
            "wo": [None for _ in range(len(layers))],
        },
        "max": {
            "q":  [None for _ in range(len(layers))],
            "k":  [None for _ in range(len(layers))],
            "v":  [None for _ in range(len(layers))],
            "wo": [None for _ in range(len(layers))],
        },
        "absmax": {
            "q":  [None for _ in range(len(layers))],
            "k":  [None for _ in range(len(layers))],
            "v":  [None for _ in range(len(layers))],
            "wo": [None for _ in range(len(layers))],
        },
    }
    if smooth_scale is not None:
        res["smooth_min"] = {
            "k":  [None for _ in range(len(layers))],
            "v":  [None for _ in range(len(layers))],
        }
        res["smooth_max"] = {
            "k":  [None for _ in range(len(layers))],
            "v":  [None for _ in range(len(layers))],
        }
        res["smooth_absmax"] = {
            "k":  [None for _ in range(len(layers))],
            "v":  [None for _ in range(len(layers))],
        }

    # insert hook to each layer
    def stat_output_hook(m:torch.nn.Module, x:torch.Tensor, y:torch.Tensor, ttype:str, layer:int):
        ''' y: module output '''

        bs, seq_len, hidden = y.shape

        for info_type in res:
            act = y.reshape(-1, hidden)
            smooth_act = act.mul(smooth_scale[ttype][layer].to(y.dtype).to(y.device)) if (("smooth" in info_type) and ttype in ["k", "v"]) else act

            if (info_type == "min" or info_type == "smooth_min"):
                # channel min
                if (("smooth" in info_type) and (not (ttype == "k" or ttype == "v"))):
                    continue

                if res[info_type][ttype][layer] is not None:
                    res[info_type][ttype][layer] = torch.min(
                        smooth_act.amin(dim=0).cpu(),
                        res[info_type][ttype][layer]
                    )
                else:
                    res[info_type][ttype][layer] = smooth_act.amin(dim=0).cpu()
            elif (info_type == "max" or info_type == "smooth_max"):
                # channel max
                if (("smooth" in info_type) and (not (ttype == "k" or ttype == "v"))):
                    continue

                if res[info_type][ttype][layer] is not None:
                    res[info_type][ttype][layer] = torch.max(
                        smooth_act.amax(dim=0).cpu(),
                        res[info_type][ttype][layer]
                    )
                else:
                    res[info_type][ttype][layer] = smooth_act.amax(dim=0).cpu()
            elif (info_type == "absmax" or info_type == "smooth_absmax"):
                # channel absmax
                if (("smooth" in info_type) and (not (ttype == "k" or ttype == "v"))):
                    continue

                if res[info_type][ttype][layer] is not None:
                    res[info_type][ttype][layer] = torch.max(
                        smooth_act.abs().amax(dim=0).cpu(),
                        res[info_type][ttype][layer]
                    )
                else:
                    res[info_type][ttype][layer] = smooth_act.abs().amax(dim=0).cpu()
            elif (info_type == "hist" or info_type == "smooth_hist"):
                if not (ttype == "k" or ttype == "v"):
                    continue

                hists=[]
                for i in range(hidden):
                    hist=torch.histc(smooth_act[:, i].to(torch.float32), bins=num_hist_bin)
                    hists.append(hist)
                if res[info_type][ttype][layer] is not None:
                    res[info_type][ttype][layer] += torch.vstack(hists).cpu()
                else:
                    res[info_type][ttype][layer] = torch.vstack(hists).cpu()
            else:
                raise ValueError(f"{info_type} not supported")

    hooks = []
    for layer_idx, layer in enumerate(layers):
        layer: LlamaDecoderLayer = layer
        layer_type = layer.__class__

        attn_act_map = {
            LlamaDecoderLayer: {
                "self_attn.q_proj": "q",
                "self_attn.k_proj": "k",
                "self_attn.v_proj": "v",
            },
            MistralDecoderLayer: {
                "self_attn.q_proj": "q",
                "self_attn.k_proj": "k",
                "self_attn.v_proj": "v",
            },
        }

        if layer_type in attn_act_map:
            for mname, module in layer.named_modules():
                if mname in attn_act_map[layer_type]:
                    hooks.append(
                        module.register_forward_hook(partial(stat_output_hook, ttype=attn_act_map[layer_type][mname], layer=layer_idx))
                    )
                if (layer_type == LlamaDecoderLayer or layer_type == MistralDecoderLayer) and mname == "self_attn.o_proj":
                    for info_type in res:
                        if info_type == "min":
                            res[info_type]["wo"] = module.weight.amin(dim=0).cpu()
                        elif info_type == "max":
                            res[info_type]["wo"] = module.weight.amax(dim=0).cpu()
                        elif info_type == "absmax":
                            res[info_type]["wo"] = module.weight.abs().amax(dim=0).cpu()
                        # elif (info_type == "hist" or info_type == "smooth_hist"):
                        #     continue
                        else:
                            continue
                            # raise ValueError(f"{info_type} not supported")
                    
    data = get_data(dataset, tokenizer, sample_len, nsample)

    # stat activation
    print("Collecting QKV scales...")
    for input_ids in tqdm(data):
        model(input_ids[0].to("cuda:0"))

    for hook in hooks:
        hook.remove()

    return res


def get_reorder_indices(
    data: dict[str, dict[str, list[torch.Tensor]]],
    n_cluster:int,
    save_path:str,
    metric:Literal["minmax", "absmax", "hist", "smooth_hist", "smooth_minmax"]="minmax",
    num_layers=32,
):
    cmin, cmax, cabsmax = data["min"], data["max"], data["absmax"]

    reorder_indices = []
    cluster_st_inds = []
    for layer in range(num_layers):
        kmin = cmin["k"][layer]
        kmax = cmax["k"][layer]
        kabsmax = cabsmax["k"][layer]
        if metric == "minmax":
            kmeans = KMeans(n_cluster, n_init=10).fit(torch.stack((kmin, kmax)).T)
        elif metric == "absmax":
            kmeans = KMeans(n_cluster, n_init=10).fit(torch.unsqueeze(kabsmax, 1))
        elif metric == "hist":
            kmeans = KMeans(n_cluster, n_init=10).fit(data["hist"]["k"][layer])
        elif metric == "smooth_hist":
            kmeans = KMeans(n_cluster, n_init=10).fit(data["smooth_hist"]["k"][layer])
        elif metric == "smooth_minmax":
            kmeans = KMeans(n_cluster, n_init=10).fit(torch.stack((data["smooth_min"]["k"][layer], data["smooth_max"]["k"][layer])).T)
        else:
            raise ValueError(f"{metric} not supported")
        k_label = torch.from_numpy(kmeans.labels_)

        k_indices = k_label.argsort()
        k_bin_count_cumsum = torch.zeros(n_cluster+1, dtype=torch.int64)
        k_bin_count_cumsum[1:] = k_label.bincount().cumsum(0).to(torch.int64)

        vmin = cmin["v"][layer]
        vmax = cmax["v"][layer]
        vabsmax = cabsmax["v"][layer]
        if metric == "minmax":
            kmeans = KMeans(n_cluster, n_init=10).fit(torch.stack((vmin, vmax)).T)
        elif metric == "absmax":
            kmeans = KMeans(n_cluster, n_init=10).fit(torch.unsqueeze(vabsmax, 1))
        elif metric == "hist":
            kmeans = KMeans(n_cluster, n_init=10).fit(data["hist"]["v"][layer])
        elif metric == "smooth_hist":
            kmeans = KMeans(n_cluster, n_init=10).fit(data["smooth_hist"]["v"][layer])
        elif metric == "smooth_minmax":
            kmeans = KMeans(n_cluster, n_init=10).fit(torch.stack((data["smooth_min"]["v"][layer], data["smooth_max"]["v"][layer])).T)
        else:
            raise ValueError(f"{metric} not supported")
        v_label = torch.from_numpy(kmeans.labels_)
        v_indices = v_label.argsort()
        v_bin_count_cumsum = torch.zeros(n_cluster+1, dtype=torch.int64)
        v_bin_count_cumsum[1:] = v_label.bincount().cumsum(0).to(torch.int64)

        reorder_indices.append((k_indices, v_indices))
        cluster_st_inds.append((k_bin_count_cumsum, v_bin_count_cumsum))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({"reorder_indices": reorder_indices, "cluster_st_inds": cluster_st_inds}, save_path)


def get_smooth_factor(
    data: dict[str, dict[str, list[torch.Tensor]]],
    save_path: str,
    alpha:float=0.6,
    percentile: float=1.0,
    num_layers:int=32,
):
    ''' smooth [Q, K], [V, O] '''
    qmax = data["absmax"]["q"]
    kmax = data["absmax"]["k"]

    vmax = data["absmax"]["v"]
    omax = data["absmax"]["wo"]

    scale = {
        "k": [],
        "v": [],
    }

    for layer in range(num_layers):
        # [seq_len, head_dim*num_heads]
        qmax_layer = qmax[layer]*percentile
        kmax_layer = kmax[layer]*percentile
        vmax_layer = vmax[layer]*percentile
        omax_layer = omax[layer]*percentile
        if alpha == 1.0:
            scale["k"].append(1/kmax_layer.pow(alpha).clamp(min=1e-5))
            scale["v"].append(1/vmax_layer.pow(alpha).clamp(min=1e-5))
        else:
            scale["k"].append((kmax_layer.pow(alpha)/qmax_layer.pow(1-alpha)).clamp(min=1e-5))
            scale["v"].append((vmax_layer.pow(alpha)/omax_layer.pow(1-alpha)).clamp(min=1e-5))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(scale, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--dataset', type=str, default="wikitext2-v1")
    parser.add_argument('--nsample', type=int, default=None)
    parser.add_argument('--sample_len', type=int, default=None)

    args = parser.parse_args()

    model_name = args.model_name

    model_to_data_size = {
        "llama-7b": (2048, 256),

        "llama2-7b": (4096, 256),
        "llama2-7b-32k": (4096, 256),
        "llama2-7b-80k": (4096, 256),
        "llama2-7b-chat": (4096, 256),
        "llama2-13b": (4096, 256),
        "llama2-13b-chat": (4096, 256),
        "llama2-70b": (4096, 256),
        "llama3-70b-instruct": (4096, 256),

        "mistral-7b": (4096, 256),
        "mistral-7b-instruct-v0.2": (4096, 256),

        "longchat-v1.5-7b-32k": (4096, 256),
        "vicuna-v1.5-7b-16k": (4096, 256),
    }

    model_path = MODEL_NAME_TO_PATH[model_name]
    sample_len, nsample = model_to_data_size[model_name]
    dataset = args.dataset

    sample_len = args.sample_len if args.sample_len is not None else sample_len
    nsample = args.nsample if args.nsample is not None else nsample

    GROUP_SIZE=[32, 64, 128]

    CACHE_DIR = f"{os.path.dirname(__file__)}/calib_cache/"

    # 1. get stat info
    stat_path = f"{CACHE_DIR}/{model_name}-{dataset}-n{nsample}-len{sample_len}.pt"

    if os.path.exists(stat_path):
        print(f"{stat_path} already existed")
    else:
        # tokenizer, model, handle = load_model_and_plug_quantizer(model_path, use_flash_attn=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        res = calibration(model, dataset, tokenizer, sample_len, nsample, smooth_scale=None)
        os.makedirs(os.path.dirname(stat_path), exist_ok=True)
        torch.save(res, stat_path)

    # 2. cal reorder indices
    model_cfg = AutoConfig.from_pretrained(model_path)
    head_dim = model_cfg.hidden_size // model_cfg.num_attention_heads
    kv_hidden = model_cfg.num_key_value_heads*head_dim
    nclusters = [kv_hidden // gsize for gsize in GROUP_SIZE]
    num_layers = model_cfg.num_hidden_layers

    metric="minmax"
    for ncluster in nclusters:
        reorder_path = f"{CACHE_DIR}/{model_name}-{dataset}-n{nsample}-len{sample_len}-{metric}-rod_idx-cluster_{ncluster}.pt"
        if os.path.exists(reorder_path):
            print(f"{reorder_path} already existed")
        else:
            print(f"cal reorder idx with {ncluster}-clusters...")
            get_reorder_indices(
                torch.load(stat_path),
                ncluster,
                reorder_path,
                metric=metric,
                num_layers=num_layers,
            )

    # # 3. get K-cache smooth scaling factor
    # for (alpha, percentile) in [(1.0, 1.0), (0.7, 0.9), (0.7, 1.0)]:
    #     ptag = "" if percentile == 1.0 else f"-p{percentile:.2f}"
    #     scale_path = f"{CACHE_DIR}/{model_name}-{dataset}-n{nsample}-len{sample_len}-smooth-alpha{alpha}{ptag}.pt"

    #     if os.path.exists(scale_path):
    #         print(f"{scale_path} already existed")
    #     else:
    #         print("cal smooth scaling factor ...")
    #         get_smooth_factor(
    #             torch.load(save_path),
    #             scale_path,
    #             alpha=alpha,
    #             percentile=percentile,
    #             num_layers=num_layers,
    #         )