import os

PROJ_DIR = f"{os.path.dirname(__file__)}"
CALIB_DIR = f"{PROJ_DIR}/calib_cache"

DATASET_CACHE = {
    "wikitext2-v1": "YOUR_PATH_TO/.cache/huggingface/datasets/wikitext/wikitext-2-raw-v1/1.0.0/a241db52902eaf2c6aa732210bead40c090019a499ceb13bcbfa3f8ab646a126",
    "ptb": "YOUR_PATH_TO/.cache/huggingface/datasets/ptb_text_only/penn_treebank/1.1.0/8d1b97746fb9765d140e569ec5ddd35e20af4d37761f5e1bf357ea0b081f2c1f",
}

MODEL_NAME_TO_PATH = {
    "llama-7b": "YOUR_PATH_TO/llama-7b",
    "llama2-7b": "YOUR_PATH_TO/llama2-7b",
    "llama2-7b-chat": "YOUR_PATH_TO/llama2-7b-chat",
    "llama2-7b-80k": "YOUR_PATH_TO/llama2-7b-80k",
    "llama2-7b-32k": "YOUR_PATH_TO/llama2-7b-32k",
    "llama2-13b": "YOUR_PATH_TO/llama2-13b",
    "llama2-13b-chat": "YOUR_PATH_TO/llama2-13b-chat",
    "llama2-70b": "YOUR_PATH_TO/Llama-2-70b-hf",
    "llama3-70b-instruct": "YOUR_PATH_TO/Meta-Llama-3-70B-Instruct",
    "mistral-7b": "YOUR_PATH_TO/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.1/snapshots/26bca36bde8333b5d7f72e9ed20ccda6a618af24",
    "mistral-7b-instruct-v0.2": "YOUR_PATH_TO/mistral-7b-instruct-v0.2",
    "vicuna-v1.5-7b-16k": "YOUR_PATH_TO/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5-16k/snapshots/c8df3ca4436a3bce5c4b5877e0117032081852b4",
    "longchat-v1.5-7b-32k": "YOUR_PATH_TO/.cache/huggingface/hub/models--lmsys--longchat-7b-v1.5-32k/snapshots/16deb633ef4d6a18d5750239edc5a85ffeaf3918",
}


MODEL_TO_SMOOTH = {
    "llama-7b": f"{CALIB_DIR}/llama-7b-wikitext2-v1-n256-len2048-smooth-alpha1.0.pt",
    "llama2-7b": f"{CALIB_DIR}/llama2-7b-wikitext2-v1-n256-len4096-smooth-alpha1.0.pt",
    "llama2-7b-chat": f"{CALIB_DIR}/llama2-7b-chat-wikitext2-v1-n256-len4096-smooth-alpha1.0.pt",
    "llama2-7b-80k": f"{CALIB_DIR}/llama2-7b-80k-wikitext2-v1-n256-len4096-smooth-alpha1.0.pt",
    # "llama2-7b-80k": f"{CALIB_DIR}/llama2-7b-80k-g128-smooth-learning-n4096.pt",
    "llama2-7b-32k": f"{CALIB_DIR}/llama2-7b-32k-wikitext2-v1-n256-len4096-smooth-alpha1.0.pt",
    # "llama2-7b-80k": f"{CALIB_DIR}/llama2-7b-80k-wikitext2-v1-n128-len20000-smooth-alpha0.7-p0.90.pt",
    # "llama2-7b-80k": f"{CALIB_DIR}/llama2-7b-80k-wikitext2-v1-n128-len20000-smooth-alpha1.0.pt",
    "llama2-13b": f"{CALIB_DIR}/llama2-13b-wikitext2-v1-n256-len4096-smooth-alpha0.7-p0.90.pt",
    "llama2-13b-chat": f"{CALIB_DIR}/llama2-13b-chat-wikitext2-v1-n256-len4096-smooth-alpha1.0.pt",
    "llama2-70b": f"{CALIB_DIR}/llama2-70b-wikitext2-v1-n256-len4096-smooth-alpha0.7-p0.90.pt",
    "llama3-70b-instruct": f"{CALIB_DIR}/llama3-70b-wikitext2-v1-n256-len4096-smooth-alpha1.0.pt",
    "mistral-7b": f"{CALIB_DIR}/mistral-7b-wikitext2-v1-n256-len4096-smooth-alpha1.0.pt",
    "mistral-7b-instruct-v0.2": f"{CALIB_DIR}/mistral-7b-instruct-v0.2-wikitext2-v1-n256-len4096-smooth-alpha1.0.pt",
    "vicuna-v1.5-7b-16k": f"{CALIB_DIR}/vicuna-v1.5-7b-16k-wikitext2-v1-n256-len4096-smooth-alpha1.0.pt",
    "longchat-v1.5-7b-32k": f"{CALIB_DIR}/longchat-v1.5-7b-32k-wikitext2-v1-n256-len4096-smooth-alpha1.0.pt",
}

MODEL_TO_REORDER = {
    "llama-7b": {
        128: {
            "minmax": f"{CALIB_DIR}/llama-7b-wikitext2-v1-n256-len2048-minmax-rod_idx-cluster_32.pt",
        },
        64: {
            "minmax": f"{CALIB_DIR}/llama-7b-wikitext2-v1-n256-len2048-minmax-rod_idx-cluster_64.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/llama-7b-wikitext2-v1-n256-len2048-minmax-rod_idx-cluster_128.pt",
        },
        16: {
            "minmax": f"{CALIB_DIR}/llama-7b-wikitext2-v1-n256-len2048-minmax-rod_idx-cluster_256.pt",
        },
    },
    "llama2-7b": {
        128: {
            "minmax": f"{CALIB_DIR}/llama2-7b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_32.pt",
        },
        64: {
            "minmax": f"{CALIB_DIR}/llama2-7b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_64.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/llama-7b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_128.pt",
        },
        16: {
            "minmax": f"{CALIB_DIR}/llama-7b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_256.pt",
        },
    },
    "llama2-7b-chat": {
        256: {
            "minmax": f"{CALIB_DIR}/llama2-7b-chat-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_16.pt",
        },
        128: {
            "minmax": f"{CALIB_DIR}/llama2-7b-chat-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_32.pt",
        },
        64: {
            "minmax": f"{CALIB_DIR}/llama2-7b-chat-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_64.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/llama2-7b-chat-wikitext2-v1-n128-len4000-minmax-rod_idx-cluster_64.pt",
        },
        16: {
            "minmax": f"{CALIB_DIR}/llama2-7b-chat-wikitext2-v1-n128-len4000-minmax-rod_idx-cluster_256.pt",
        },
    },
    "llama2-13b-chat": {
        128: {
            "minmax": f"{CALIB_DIR}/llama2-13b-chat-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_40.pt",
        },
        64: {
            "minmax": f"{CALIB_DIR}/llama2-13b-chat-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_80.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/llama2-13b-chat-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_160.pt",
        },
    },
    "llama2-13b": {
        128: {
            "minmax": f"{CALIB_DIR}/llama2-13b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_40.pt",
        },
        64: {
            "minmax": f"{CALIB_DIR}/llama2-13b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_80.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/llama2-13b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_160.pt",
        },
    },
    "llama2-7b-80k": {
        128: {
            "minmax": f"{CALIB_DIR}/llama2-7b-80k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_32.pt",
        },
        64: {
            "minmax": f"{CALIB_DIR}/llama2-7b-80k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_64.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/llama2-7b-80k-wikitext2-v1-n128-len20000-minmax-rod_idx-cluster_128.pt",
        },
    },
    "llama2-7b-32k": {
        128: {
            "minmax": f"{CALIB_DIR}/llama2-7b-32k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_32.pt",
        },
        64: {
            "minmax": f"{CALIB_DIR}/llama2-7b-32k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_64.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/llama2-7b-32k-wikitext2-v1-n128-len20000-minmax-rod_idx-cluster_128.pt",
        },
    },
    "llama2-70b": {
        128: {
            "minmax": f"{CALIB_DIR}/llama2-70b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_8.pt",
        },
        64: {
            "minmax": f"{CALIB_DIR}/llama2-70b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_16.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/llama2-70b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_32.pt",
        },
    },
    "llama3-70b-instruct": {
        128: {
            "minmax": f"{CALIB_DIR}/llama3-70b-instruct-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_8.pt",
        },
        64: {
            "minmax": f"{CALIB_DIR}/llama3-70b-instruct-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_16.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/llama3-70b-instruct-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_32.pt",
        },
    },
    "mistral-7b": {
        256: {
            "minmax": f"{CALIB_DIR}/mistral-7b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_4.pt"
        },
        128: {
            "minmax": f"{CALIB_DIR}/mistral-7b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_8.pt"
        },
        64: {
            "minmax": f"{CALIB_DIR}/mistral-7b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_16.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/mistral-7b-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_32.pt",
        },
    },
    "mistral-7b-instruct-v0.2": {
        256: {
            "minmax": f"{CALIB_DIR}/mistral-7b-instruct-v0.2-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_4.pt"
        },
        128: {
            "minmax": f"{CALIB_DIR}/mistral-7b-instruct-v0.2-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_8.pt"
        },
        64: {
            "minmax": f"{CALIB_DIR}/mistral-7b-instruct-v0.2-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_16.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/mistral-7b-instruct-v0.2-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_32.pt",
        },
    },
    "vicuna-v1.5-7b-16k": {
        256: {
            "minmax": f"{CALIB_DIR}/vicuna-v1.5-7b-16k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_16.pt",
        },
        128: {
            "minmax": f"{CALIB_DIR}/vicuna-v1.5-7b-16k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_32.pt",
        },
        64: {
            "minmax": f"{CALIB_DIR}/vicuna-v1.5-7b-16k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_64.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/vicuna-v1.5-7b-16k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_128.pt",
        },
    },
    "longchat-v1.5-7b-32k": {
        256: {
            "minmax": f"{CALIB_DIR}/longchat-v1.5-7b-32k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_16.pt",
        },
        128: {
            "minmax": f"{CALIB_DIR}/longchat-v1.5-7b-32k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_32.pt",
        },
        64: {
            "minmax": f"{CALIB_DIR}/longchat-v1.5-7b-32k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_64.pt",
        },
        32: {
            "minmax": f"{CALIB_DIR}/longchat-v1.5-7b-32k-wikitext2-v1-n256-len4096-minmax-rod_idx-cluster_128.pt",
        },
    },
}
