"""
CUDA_VISIBLE_DEVICES=0 python eval_needle.py \
    --model_name llama2-7b-80k \
    --quant k2-v2-w128-g128-reorder-pre_rope-clip-sink5-fp8 \
    --ctx_len 32000 \
"""

import argparse
import logging
import os
import glob
import json
import time
import torch
import numpy as np
from typing import Literal
from datetime import datetime, timezone
from fastchat.model import get_conversation_template
from transformers import AutoTokenizer

# from rkvq.quant.quantizer import FakeQuantizer
# from rkvq.quant.quant_llama import load_model_and_plug_quantizer
# from quant_config import get_quantizer
from KVcache_manager import ModelKVCacheManager
from experiments.modeling_llama_skvq import LlamaForCausalLM
from experiments.utils import plug_quantizer_into_model, rouge1, rougeL
from calib_config import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", required=True)
parser.add_argument("--quant", type=str, default=None)
parser.add_argument(
    "--ctx_len", required=False, type=int, default=32000
)  # default is 32k

# group quant
parser.add_argument("--group", default=None, type=int, choices=[32, 64, 128])
parser.add_argument("--kbits", default=None, choices=[1, 1.5, 2])
parser.add_argument("--vbits", default=None, choices=[1, 1.5, 2])
# POQ window
parser.add_argument("--window", default=128, type=int, choices=[32, 64, 128])
# reorder
parser.add_argument(
    "--krod_ncluster", default=None, type=int, choices=[128, 64, 32, 16]
)
parser.add_argument(
    "--vrod_ncluster", default=None, type=int, choices=[128, 64, 32, 16]
)
parser.add_argument(
    "--krod_metric", default="minmax", type=str, choices=["minmax", "absmax", "hist"]
)
parser.add_argument(
    "--vrod_metric", default="minmax", type=str, choices=["minmax", "absmax", "hist"]
)
parser.add_argument("--krod_percentile", default=None, type=float)
parser.add_argument("--vrod_percentile", default=None, type=float)
# # nuq
# parser.add_argument("--knuq", default=None, type=float)
# parser.add_argument("--vnuq", default=None, type=float)


logger = logging.getLogger(__name__)


PROMPT_TEMPLATE = """
You are a helpful AI bot that answers questions for a user. Keep your response short and direct

Human: <context>
{context}
</context>

{retrieval_question} Don't give information outside the document or repeat your findings

Assistant: Here is the most relevant sentence in the context:
"""

DATA_DIR = f"{os.path.dirname(__file__)}/dataset/PaulGrahamEssays"


def reset_rope(model, model_max_train_len, scaling_factor):
    for l in model.model.layers:
        l.self_attn.rotary_emb.scaling_factor = scaling_factor
        l.self_attn.rotary_emb._set_cos_sin_cache(
            seq_len=model_max_train_len, device="cuda", dtype=torch.float32
        )
    return


class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """

    def __init__(
        self,
        needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        haystack_dir=DATA_DIR,
        retrieval_question="What is the best thing to do in San Francisco?",
        results_version=1,
        context_lengths_min=500,
        context_lengths_max=3500,
        context_lengths_num_intervals=35,
        context_lengths=None,
        document_depth_percent_min=0,
        document_depth_percent_max=100,
        document_depth_percent_intervals=35,
        document_depth_percents=None,
        document_depth_percent_interval_type="linear",
        model_path="YOUR_PATH_TO/models/llama2-7b-chat",
        fake_quantizer: ModelKVCacheManager = None,
        model_name="llama2-7b-chat-4k",
        test_model=None,
        num_concurrent_requests=1,
        save_results=True,
        save_contexts=True,
        final_context_length_buffer=200,
        save_dir="./needle_results",
        print_ongoing_status=True,
    ):
        """
        :param needle: The needle to be found in the haystack. Default is None.
        :param haystack_dir: The directory of text files to use as background context (or a haystack) in which the needle is to be found. Default is Paul Graham Essays.
        :param retrieval_question: The question which with to prompt the model to do the retrieval.
        :param results_version: In case you would like to try the same combination of model, context length, and depth % multiple times, change the results version other than 1
        :param num_concurrent_requests: Due to volume, this object is set up to run concurrent requests, default = 1. Be careful of rate limits.
        :param save_results: Whether or not you would like to save your contexts to file. Warning: These will get long! Default = True
        :param save_contexts: Whether or not you would like to save your contexts to file. Warning: These will get long! Default is True.
        :param final_context_length_buffer: The amount of cushion you'd like to leave off the input context to allow for the output context. Default 200 tokens
        :param context_lengths_min: The minimum length of the context. Default is 1000.
        :param context_lengths_max: The maximum length of the context. Default is 200000.
        :param context_lengths_num_intervals: The number of intervals for the context length. Default is 35.
        :param context_lengths: The lengths of the context. Default is None.
        :param document_depth_percent_min: The minimum depth percent of the document. Default is 0.
        :param document_depth_percent_max: The maximum depth percent of the document. Default is 100.
        :param document_depth_percent_intervals: The number of intervals for the document depth percent. Default is 35.
        :param document_depth_percents: The depth percentages of the document. Default is None.
        :param document_depth_percent_interval_type: The type of interval for the document depth percent. Must be either 'linear' or 'sigmoid'. Default is 'linear'.
        :param model_path: model weight path.
        :param openai_api_key: The API key for OpenAI. Default is None.
        :param anthropic_api_key: The API key for Anthropic. Default is None.
        :param model_name: The name of the model. Default is 'gpt-4-1106-preview'.
        :param print_ongoing_status: Whether or not to print the ongoing status. Default is True.
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError(
                "Needle, haystack, and retrieval_question must be provided."
            )

        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.print_ongoing_status = print_ongoing_status
        self.model_path = model_path
        self.testing_results = []
        self.save_dir = save_dir

        # For Evaluation
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if context_lengths is None:
            if (
                context_lengths_min is None
                or context_lengths_max is None
                or context_lengths_num_intervals is None
            ):
                raise ValueError(
                    "Either context_lengths_min, context_lengths_max, context_lengths_intervals need to be filled out OR the context_lengths_list needs to be supplied."
                )
            else:
                self.context_lengths = np.round(
                    np.linspace(
                        context_lengths_min,
                        context_lengths_max,
                        num=context_lengths_num_intervals,
                        endpoint=True,
                    )
                ).astype(int)
        else:
            self.context_lengths = context_lengths

        if document_depth_percents is None:
            if (
                document_depth_percent_min is None
                or document_depth_percent_max is None
                or document_depth_percent_intervals is None
            ):
                raise ValueError(
                    "Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied."
                )
            else:
                if document_depth_percent_interval_type == "linear":
                    self.document_depth_percents = np.round(
                        np.linspace(
                            document_depth_percent_min,
                            document_depth_percent_max,
                            num=document_depth_percent_intervals,
                            endpoint=True,
                        )
                    ).astype(int)
                elif document_depth_percent_interval_type == "sigmoid":
                    self.document_depth_percents = [
                        self.logistic(x)
                        for x in np.linspace(
                            document_depth_percent_min,
                            document_depth_percent_max,
                            document_depth_percent_intervals,
                        )
                    ]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError(
                "document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals"
            )

        self.model_name = model_name

        if fake_quantizer:
            fake_quantizer.active(True)
        #     use_flash_attn = False
        # else:
        #     use_flash_attn = True
        self.enc = AutoTokenizer.from_pretrained(model_path, use_fast=False, trust_remote_code=True)
        if test_model is not None:
            self.model_to_test = test_model
        else:
            self.model_to_test = LlamaForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                torch_dtype=torch.float16,
                use_flash_attention_2=True,
            )
        plug_quantizer_into_model(self.model_to_test, fake_quantizer)
        self.fake_quantizer = fake_quantizer
        # self.enc, self.model_to_test, self.quantizer_handle = load_model_and_plug_quantizer(
        #     model_path, fake_quantizer=fake_quantizer, parallisim="tp", use_flash_attn=use_flash_attn)

        # self.enc = AutoTokenizer.from_pretrained(model_path)
        # self.model_to_test = AutoModelForCausalLM.from_pretrained(model_path,
        #                                                         use_flash_attention_2="flash_attention_2",
        #                                                         torch_dtype=torch.bfloat16,
        #                                                         ).cuda().eval()
        # self.model_to_test = tp.tensor_parallel(self.model_to_test, sharded=True)

        if "llama2-7b-80k" in model_name:
            scaling_factor = 10  # hardcode
            reset_rope(
                self.model_to_test,
                model_max_train_len=81920,
                scaling_factor=scaling_factor,
            )

        self.model_to_test_description = model_name
        # self.evaluation_model = ChatOpenAI(model="gpt-4", temperature=0, openai_api_key = self.openai_api_key)

    def logistic(self, x, L=100, x0=50, k=0.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def bound_evaluate_and_log(self, *args):
        self.evaluate_and_log(*args)

    def run_test(self):
        # Run through each iteration of context_lengths and depths
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                self.bound_evaluate_and_log(context_length, depth_percent)

    def generate_prompt(self, context):
        # align to https://github.com/FranxYao/Long-Context-Data-Engineering/blob/main/eval/needle/needle_in_haystack.py
        if "llama2-7b-80k" in self.model_name:
            test_format = f"<|im_start|> This is a very long story book: <book> {context} </book>.\n Based on the content of the book, Question: {self.retrieval_question}\nAnswer:"
            return test_format
        elif "longchat" in self.model_name or "vicuna" in self.model_name:
            conv = get_conversation_template("vicuna")
            conv.append_message(
                conv.roles[0],
                f"You are a helpful AI bot that answers questions for a user. Keep your response short and direct.\nHuman: <context>{context}</context> {self.retrieval_question} Don't give information outside the document or repeat your findings)",
            )
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
        else:
            return PROMPT_TEMPLATE.format(
                retrieval_question=self.retrieval_question, context=context
            )

    def evaluate_and_log(self, context_length, depth_percent):
        # Checks to see if you've already checked a length/percent/version.
        # This helps if the program stop running and you want to restart later
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                return

        # Go generate the required length context and place your needle statement in
        context = self.generate_context(context_length, depth_percent)

        # Prepare your message to send to the model you're going to evaluate
        prompt = self.generate_prompt(context)

        prompt_ids = self.enc.encode(prompt, return_tensors="pt")
        prompt_tok_len = prompt_ids.shape[-1]
        # print([prompt])
        # logger.info(f"{[prompt]}")
        test_start_time = time.perf_counter()
        with torch.no_grad():
            response_ids = self.model_to_test.generate(
                prompt_ids.to(self.model_to_test.device),
                max_new_tokens=60,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=self.enc.eos_token_id,
            )
            response = self.enc.decode(
                response_ids[0][prompt_tok_len:], skip_special_tokens=True
            )
        test_end_time = time.perf_counter()
        test_elapsed_time = test_end_time - test_start_time
        torch.cuda.empty_cache()

        # Compare the reponse to the actual needle you placed
        score = self.evaluate_response(response, "ROUGE")

        results = {
            # 'context' : context, # Uncomment this line if you'd like to save the context the model was asked to retrieve from. Warning: This will become very large.
            "model": self.model_to_test_description,
            "context_length": int(context_length),
            "depth_percent": float(depth_percent),
            "version": self.results_version,
            "needle": self.needle,
            "model_response": response,
            "score": score,
            "test_duration_seconds": test_elapsed_time,
            "test_timestamp_utc": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S%z"
            ),
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            logger.info(f"-- Test Summary -- ")
            logger.info(f"Duration: {test_elapsed_time:.1f} seconds")
            logger.info(f"Context: {context_length} tokens")
            logger.info(f"Depth: {depth_percent}%")
            logger.info(f"Score: {score}")
            logger.info(f"Response: {[response]}\n")

        context_file_location = f'{self.model_name.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent)}'

        if self.save_contexts:
            results["file_name"]: context_file_location

            # Save the context to file for retesting
            if not os.path.exists("contexts"):
                os.makedirs("contexts")

            with open(f"contexts/{context_file_location}_context.txt", "w") as f:
                f.write(context)

        if self.save_results:
            # Save the context to file for retesting
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

            # Save the result to file for retesting
            with open(
                f"{self.save_dir}/{context_file_location}_results.json", "w"
            ) as f:
                json.dump(results, f)

    def result_exists(self, context_length, depth_percent):
        """
        Checks to see if a result has already been evaluated or not
        """

        results_dir = "results/"
        if not os.path.exists(results_dir):
            return False

        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir, filename), "r") as f:
                    result = json.load(f)
                    context_length_met = result["context_length"] == context_length
                    depth_percent_met = result["depth_percent"] == depth_percent
                    version_met = result.get("version", 1) == self.results_version
                    model_met = result["model"] == self.model_name
                    if (
                        context_length_met
                        and depth_percent_met
                        and version_met
                        and model_met
                    ):
                        return True
        return False

    def generate_context(self, context_length, depth_percent):
        # Load up tiktoken so we navigate tokens more easily

        # Get your Paul Graham files loaded into a string
        context = self.read_context_files()

        # Truncate the Paul Graham essays to the context length you desire
        context = self.encode_and_trim(context, context_length)

        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def encode_text_to_tokens(self, text):
        return self.enc.encode(text)

    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[: context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            # period_tokens = self.encode_text_to_tokens('.')

            period_tokens = [869, 29889]  # hard-code for llama and internlm2 tokenizer
            if "internlm2" in self.model_name:
                period_tokens = [790, 281]

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            logger.info(f"insertion at {insertion_point}")
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def evaluate_response(
        self, response: str, evaluator: Literal["GPT4", "ROUGE"] = "GPT4"
    ):
        if evaluator == "GPT4":
            pass
            # accuracy_criteria = {
            #     "accuracy": """
            #     Score 1: The answer is completely unrelated to the reference.
            #     Score 3: The answer has minor relevance but does not align with the reference.
            #     Score 5: The answer has moderate relevance but contains inaccuracies.
            #     Score 7: The answer aligns with the reference but has minor omissions.
            #     Score 10: The answer is completely accurate and aligns perfectly with the reference.
            #     Only respond with a numberical score
            #     """
            # }

            # # Using GPT-4 to evaluate
            # evaluator = load_evaluator(
            #     "labeled_score_string",
            #     criteria=accuracy_criteria,
            #     llm=self.evaluation_model,
            # )

            # eval_result = evaluator.evaluate_strings(
            #     # The models response
            #     prediction=response,

            #     # The actual answer
            #     reference=self.needle,

            #     # The question asked
            #     input=self.retrieval_question,
            # )

            # return int(eval_result['score'])
        elif evaluator == "ROUGE":
            ref = self.enc.encode(self.needle.strip(), add_special_tokens=False)
            pred = self.enc.encode(response.strip(), add_special_tokens=False)
            rouge1_f1 = rouge1(ref, pred)[-1]
            rougeL_f1 = rougeL(ref, pred)[-1]
            return {"rouge1_f1": rouge1_f1, "rougeL_f1": rougeL_f1}
        else:
            raise ValueError(f"{evaluator} is not supported")

    def get_context_length_in_tokens(self, context):
        return len(self.enc.encode(context))

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, "r") as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        return self.enc.encode(context)

    def decode_tokens(self, tokens, context_length=None):
        # return self.enc.decode(tokens[:context_length], skip_special_tokens=True)
        return self.enc.decode(tokens[:context_length])

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context

    def get_results(self):
        return self.testing_results

    def print_start_test_summary(self):
        logger.info("\n")
        logger.info("Starting Needle In A Haystack Testing...")
        logger.info(f"- Model: {self.model_name}")
        logger.info(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}"
        )
        logger.info(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%"
        )
        logger.info(f"- Needle: {self.needle.strip()}")
        logger.info("\n\n")

    def start_test(self):
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test()


def get_quantizer_from_str(
    s: str, model: LlamaForCausalLM, model_name: str
) -> ModelKVCacheManager|None:
    """
    example: "k2-v2-g128-w128-reoder-pre_rope"
             "k2-v2-g128-w128-reoder-clip-pre_rope"
             "k2-v2-g128-w128-reoder-clip-sink-pre_rope"
             "k2-v2-g128-w128"
             "k2-v2-g128-w128-KIVI"
             "k2-v2-g128-rptq"
             "k2-v2-g128-smoothquant"
             "k2-v2-g128-rtn"
             "k2-v2-g128-h2o0.95"
             "k2-v2-g128-random0.95"
    """
    if s is None or s.strip().lower() == "none":
        return None

    opts = s.strip().split("-")
    kbits = float(opts[0][1:])
    vbits = float(opts[1][1:])
    kbits = int(kbits) if kbits == round(kbits) else kbits
    vbits = int(vbits) if vbits == round(vbits) else vbits
    gsize = int(opts[2][1:])
    if ("rtn" in s) or ("rptq" in s) or ("smoothquant" in s):
        window = 0
    else:
        window = int(opts[3][1:])

    smooth_file = MODEL_TO_SMOOTH[model_name] if "smooth" in s else None
    reorder_file = (
        MODEL_TO_REORDER[model_name][gsize]["minmax"]
        if (("reorder" in s) or ("rod" in s) or ("rptq" in s))
        else None
    )

    pre_rope = (
        True
        if ("pre_rope" in s) or ("rptq" in s) or ("smoothquant" in s)
        else False
    )

    clipping = [(0.92 if "clip" in s else 1.0) for _ in range(len(model.model.layers))] 
    full_prefill = False if ("rtn" in s) or ("rptq" in s) or ("smoothquant" in s) else True
    KIVI_mode = "KIVI" in s
    fp8 = "fp8" in s
    sink = int(s.split("sink")[1].split("-")[0]) if "sink" in s else 0
    use_acc_score = float(s.split("h2o")[1].split("-")[0]) if "h2o" in s else 0
    use_random = float(s.split("random")[1].split("-")[0]) if "random" in s else 0

    quantizer = ModelKVCacheManager.create(
        model=model,
        kbits=kbits,
        vbits=vbits,
        gsize=gsize,
        window_size=window,
        reorder_file=reorder_file,
        smooth_file=smooth_file,
        clipping=clipping,
        pre_rope=pre_rope,
        full_prefill=full_prefill,
        KIVI_mode=KIVI_mode,
        fp8=fp8,
        attn_sink=sink,
        use_acc_score=use_acc_score,
        use_random=use_random,
    )
    print(f"{'='*30}ModelKVManager{'='*30}\n{quantizer}")
    return quantizer



def main(args):
    scheme = args.quant
    model_name = args.model_name

    model_path: str = MODEL_NAME_TO_PATH[model_name]

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        use_flash_attention_2=True,
    )

    if args.quant is not None:
        fake_quantizer = get_quantizer_from_str(args.quant, model, model_name)

    context_lengths_max = args.ctx_len
    if context_lengths_max > 16000:
        context_lengths_num_intervals = 20
    else:
        context_lengths_num_intervals = 15

    save_tag = fake_quantizer.tag() if fake_quantizer else "None"

    ht = LLMNeedleHaystackTester(
        model_name=f"{model_name}-{scheme}",
        model_path=model_path,
        fake_quantizer=fake_quantizer,
        test_model=model,
        context_lengths_min=1000,
        context_lengths_max=context_lengths_max,
        context_lengths_num_intervals=context_lengths_num_intervals,
        document_depth_percent_intervals=15,
        save_contexts=False,
        save_dir=f"./needle_results/{model_name}/{save_tag}-{context_lengths_max}",
    )

    ht.start_test()


if __name__ == "__main__":
    if not os.path.exists(f"{PROJ_DIR}/needle_results"):
        os.makedirs(f"{PROJ_DIR}/needle_results")

    args = parser.parse_args()

    # ts = time.strftime("%m%d%H%M%S", time.localtime())
    # file_handler = logging.FileHandler(f"log/{args.model_name}-{args.quant}.log", "w")
    # file_handler.setFormatter(logging.Formatter("%(levelname)s - %(filename)s:%(lineno)d - %(message)s"))
    # quantizer_logger = logging.getLogger("rkvq.quant.quantizer")
    # quantizer_logger.addHandler(file_handler)
    # quantizer_logger.setLevel(logging.DEBUG)

    needle_log_dir = f"{PROJ_DIR}/log/needle/"
    os.makedirs(needle_log_dir, exist_ok=True)
    logger.addHandler(
        logging.FileHandler(
            f"{needle_log_dir}{args.model_name}-{args.quant}-{args.ctx_len}.log",
            "w",
        )
    )
    logger.setLevel(logging.INFO)

    main(args)
