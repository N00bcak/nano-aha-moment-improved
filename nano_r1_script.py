import argparse
import ast
import operator
import gc
import json
import logging
import os
import re
import shutil
import socket
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
import wandb
from datasets import Dataset, load_dataset
from deepspeed import DeepSpeedEngine
from deepspeed.runtime.utils import see_memory_usage
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams

# Set environment variable for VLLM
os.environ["VLLM_USE_V1"] = "0"

# -----------------------------------------------------------------------------
# Little gizmos to make code more readable
# -----------------------------------------------------------------------------
def process_rank() -> int:
    """Get the rank of the current process in distributed training."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0

def world_size() -> int:
    """Get the total number of processes in distributed training."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1

def is_main_process() -> bool:
    """Check if the current process is the main process (rank 0)."""
    return process_rank() == 0

def dist_barrier(device_id: Optional[int] = None):
    """
    TBD: What does this specifically do?
    """
    if not dist.is_initialized():
        return
    
    if device_id is None and torch.cuda.is_available():
        device_id = torch.cuda.current_device()
        
    if device_id is None:
        dist.barrier()
    else:
        dist.barrier(device_ids=[device_id])

# Per-process logger for training run.
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# -----------------------------------------------------------------------------
# RLVR Task Abstraction (With Countdown as its realization)
# -----------------------------------------------------------------------------

@dataclass(frozen = True)
class CountdownTask:
    """
    Ingests a dataset, preprocesses it, and provides methods for reward calculation.
    """
    system_message = (
        "You are a helpful assistant. "
        "You first think about the reasoning process in the mind "
        "and then provide the user with the answer."
    )
    prompt_template = (
        "Using the numbers {numbers}, create an equation that equals {target}. "
        "You can use basic arithmetic operations (+, -, *, /) "
        "and each number can only be used once. "
        "Show your work in <think> </think> tags. "
        "And return the final equation and answer in <answer> </answer> tags, "
        "for example <answer>(1 + 2) / (3 * 5)</answer>."
    )

    '''
    Since LLMs generate text, it can be challenging to elicit specific 
    close-ended answers which can be reliably and automatically evaluated.
    There are many strategies to deal with this, but a common approach
    is to enforce a response format for the whole LLM generation...
    '''
    _response_regex = re.compile(
        r"^<think>([^<]*(?:<(?!/?think>)[^<]*)*)<\/think>\s*<answer>([\s\S]*?)<\/answer>$"
    )
    # ... as well as the content inside the answer box.
    _answer_content_regex = re.compile(
        r"^[\d+\-*/().\s]+$"
    )

    def _normalize_response(
        self,
        response: str,
    ):
        """
        Once the model offers its response,
        we need to remove any artifacts that could complicate
        our formulaic reward calculation.

        Things like capital letters, unnecessary punctuations,
        the missing <think> tag, etc.

        We don't much care since evaluating the math
        won't be affected by errant spaces or upper-cases,
        and punctuations ARE considered illegal anyway.
        Again, something you can change based on your needs.
        """
        final_resp = response.strip().lower()
        if not final_resp.startswith("<think>"):
            final_resp = "<think>" + final_resp
        return final_resp
    
    def _format_reward(
        self,
        response: str,
    ):
        """
        For our convenience we enforce a macro response format
        of <think>...</think>\n<answer>...</answer>.
        Ideally, the answer inside also has a restricted format,
        so we check for that as well.

        Note that the rewards are min-max normalized to [0, 1],
        so we can scale them later.
        """
        response = self._normalize_response(response)

        # Check if the macro format is correct
        match = self._response_regex.search(response)
        if match is None or len(match.groups()) != 2:
            # Format is completely off
            return 0.0
        else:
            # Check if the contents of <answer>...</answer> 
            # are well-formed
            answer_content = match.group(2).strip()

            if not self._answer_content_regex.match(answer_content):
                return 0.5
            else:
                return 1.0
    
    # This is a stub that evaluates a mathematical expression safely
    # WITHOUT the use of eval.
    # Not relevant to the main point of this script,
    # so you can safely ignore it.
    _VALID_OPERATIONS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }
    def _evaluate_equation(
        self,
        equation: str,
    ) -> Tuple[bool, float]:
        def helper(valid: bool, token: str) -> Tuple[bool, float]:
            if not valid:
                return False, 0.0
            elif (
                isinstance(token, ast.Constant) 
                and isinstance(token.value, (int, float))
            ):
                return True, float(token.value)
            elif (
                isinstance(token, ast.BinOp) 
                and type(token.op) in self._VALID_OPERATIONS
            ):
                left_valid, left_value = helper(valid, token.left)
                right_valid, right_value = helper(valid, token.right)
                
                if not (left_valid and right_valid):
                    return False, 0.0
                
                return True, float(self._VALID_OPERATIONS[type(token.op)](
                    left_value, right_value
                ))
            elif (
                isinstance(token, ast.UnaryOp)
                and type(token.op) in self._VALID_OPERATIONS
            ):
                op_valid, op_operand = helper(valid, token.operand)

                if not op_valid:
                    return False, 0.0
                
                return True, float(self._VALID_OPERATIONS[type(token.op)](
                    op_operand
                ))
            else:
                return False, 0.0
        return helper(True, ast.parse(equation, mode="eval").body)        
    
    def _verify_response(
        self,
        response: str,
        nums: List[int],
        target: int,
        float_tol: float = 1e-5,
    ):
        """
        Evaluates whether the response is a valid expression
        which evaluates to the target number using the provided numbers.

        Ditto the above, rewards are min-max normalized to [0, 1].
        """
        response = self._normalize_response(response)

        # The response is contained within the <answer>...</answer> tags,
        # or the second group of the regex match.
        match = self._response_regex.search(response)
        if match is None or len(match.groups()) != 2:
            return 0.0
        
        equation = match.group(2).strip()

        # If the equation is not well-formed, we cannot evaluate it.
        # Presume it is wrong.
        if not self._answer_content_regex.match(equation):
            return 0.0
        
        # If the equation doesn't use numbers from the provided set,
        # it is automatically wrong.
        used_numbers = [int(n) for n in re.findall(r"\d+", equation)]
        if sorted(used_numbers) != sorted(nums):
            return 0.0
        
        # Yes, you COULD use `eval` here, but the output is from
        # an LLM, which can be dangerous.
        # Call me a boomer.
        # Just use an AST to parse, please...
        is_valid, result = self._evaluate_equation(equation)

        # Another note: Countdown does not have any known
        # heuristics. So we only disburse the reward if it is
        # correct (up to floating point tolerance).
        if is_valid and abs(result - float(target)) < float_tol:
            return 1.0
        else:
            return 0.0
        
    # Writing this method requires some knowledge of the dataset
    # we are working with.
    # Here we see there is a "nums" field and a "target" field,
    # which correspond to the numbers provided and the target number
    # to reach, respectively.
    # ---
    # Beyond that, we just apply a simple chat template,
    # with all the things that the LLM would expect to see
    # when it is being prompted to solve a Countdown problem.

    @staticmethod
    def _preprocess_samples(
        batch: Dict[str, Any],
        system_message: str,
        prompt_template: str,
        tokenizer: AutoTokenizer,
    ):
        nums_batch: List[List[int]] = batch["nums"]
        target_batch: List[int] = batch["target"]

        prompts: List[str] = []
        input_ids_batch: List[List[int]] = []

        for numbers, target in zip(nums_batch, target_batch):
            chat_prefix = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt_template.format(numbers=numbers, target=target)},
                {"role": "assistant", "content": "Let me solve this step by step.\n<think>"},
            ]
            input_ids = tokenizer.apply_chat_template(
                chat_prefix, tokenize=True, continue_final_message=True
            )
            prompt = tokenizer.decode(
                input_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompts.append(prompt)
            input_ids_batch.append(input_ids)

        return {"prompt": prompts, "input_ids": input_ids_batch}
    
    @staticmethod
    def load_and_preprocess_dataset(
        dataset_name: str,
        tokenizer: AutoTokenizer,

    ) -> Tuple[Dataset, Dataset]:
        """
        Load and preprocess the dataset.
        This function abstracts the dataset loading and preprocessing logic.
        """ 
        dataset = load_dataset(dataset_name, split="train")

        # Rank 0 will preprocess, others will wait and load from cache
        if dist.is_initialized() and is_main_process():
             pass 
        elif dist.is_initialized():
             dist.barrier()
        
        # The fast HuggingFace tokenizers are already parallelized,
        # so there is no need to go around and cause process thrashing
        # by using multiple processes per rank.
        # Moreover, batching is also more efficient.
        dataset = dataset.map(
            CountdownTask._preprocess_samples,
            batched = True,
            num_proc = 1,
            fn_kwargs = {
                "system_message": CountdownTask.system_message,
                "prompt_template": CountdownTask.prompt_template,
                "tokenizer": tokenizer,
            },
            desc = "Preprocessing dataset",
            load_from_cache_file = True,
        )


        dist.barrier()

        # Split the dataset into train and test sets
        train_test_split = dataset.train_test_split(test_size = 2_000, seed = 42)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
        return train_dataset, test_dataset
    
    def get_reward(
        self,
        response: str,
        sample: Dict[str, Any],
    ):
        nums = sample["nums"]
        target = sample["target"]

        format_reward = self._format_reward(response)
        verification_reward = self._verify_response(response, nums, target)

        # It is common to use a weighted sum of the two rewards.
        reward = format_reward + verification_reward

        metrics = {
            "format_reward": format_reward,
            "verification_reward": verification_reward,
        }

        return reward, metrics

# -----------------------------------------------------------------------------
# GRPO Optimization Algorithm
# -----------------------------------------------------------------------------
class GRPOAlgorithm:
    """
    Handles GRPO-specific logic such as:
    - Processing Episodes (grouping generations, computing rewards, advantages)
    - Computing GRPO Loss
    - Performing Backpropagation Step

    Cater for both reference and reference-free GRPO.
    """
    def __init__(
        self,
        policy_model: Union[DeepSpeedEngine, PreTrainedModel],
        reference_model: Optional[Union[DeepSpeedEngine, PreTrainedModel]],
        tokenizer: AutoTokenizer,
        reward_func: Callable[[str, Dict[str, Any]], Tuple[float, Dict[str, float]]],
        args: argparse.Namespace,
    ):
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.reward_func = reward_func
        self.args = args

        if args.use_ref_based_rlvr:
            logger.info("KL penalty desired; using reference-based GRPO.")
        else:
            logger.info("Using reference-free variant of GRPO.")
    
    @staticmethod
    @torch.compile(dynamic=True)
    def log_softmax_and_gather(logits: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """
        torch compiled version of the common `log_softmax -> gather` operation.
        """
        logprobs = logits.log_softmax(dim=-1)
        return torch.gather(logprobs, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def compute_token_log_probs(
        model: Union[DeepSpeedEngine, PreTrainedModel],
        inputs: Dict[str, torch.Tensor],
        temperature: float,
    ) -> torch.Tensor:
        """
        Compute log probabilities for each token in the sequence, masked for valid labels only.
        """
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            use_cache=False,
        )

        logits = outputs.logits / temperature  # Shape: [batch_size, seq_len, vocab_size]
        shift_logits = logits[..., :-1, :]  # Shape: [batch_size, seq_len-1, vocab_size]
        shift_labels = inputs["labels"][..., 1:]  # Shape: [batch_size, seq_len-1]
        shift_labels_mask = inputs["labels_mask"][..., 1:]  # Shape: [batch_size, seq_len-1]

        # Create mask for valid labels
        shift_labels[~(shift_labels_mask.bool())] = 0  # Shape: [batch_size, seq_len-1]

        # Calculate log probabilities
        log_probs = GRPOAlgorithm.log_softmax_and_gather(shift_logits, shift_labels)  # Shape: [batch_size, seq_len-1]
        log_probs = log_probs * shift_labels_mask  # Shape: [batch_size, seq_len-1]

        return log_probs

    def process_episodes(
        self,
        *,
        samples: List[Dict[str, Any]],
        all_generations: List[List[int]],
        all_finish_reasons: List[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        
        """
        Process model generations and calculate rewards for GRPO training episodes.
        """

        assert len(all_generations) == len(all_finish_reasons), (
            f"Length mismatch: {len(all_generations)} generations vs "
            f"{len(all_finish_reasons)} episodes reported."
        )
        assert len(all_generations) == len(samples) * self.args.gens_per_sample, (
            f"Length mismatch: {len(all_generations)} generations vs "
            f"{len(samples)} samples with {self.args.gens_per_sample} generations each."
        )

        groups = [
            list(range(i, i + self.args.gens_per_sample)) 
            for i in range(0, len(all_generations), self.args.gens_per_sample)
        ]
        all_query_token_ids, all_responses_token_ids, all_advantages = [], [], []

        stats = {
            "response_lengths": [],
            "rewards": [],
            "non_stop_rate": [],
        }

        for sample, group_indices in zip(samples, groups):
            response_token_ids = [all_generations[i] for i in group_indices]
            finish_reasons = [all_finish_reasons[i] for i in group_indices]
            responses = self.tokenizer.batch_decode(response_token_ids, skip_special_tokens=False)
            rewards_and_metrics = [self.reward_func(resp, sample) for resp in responses]
            rewards, reward_metrics = zip(*rewards_and_metrics)

            rewards = np.array(rewards)
            
            # 
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

            per_token_advantages = [[adv] * len(resp) for adv, resp in zip(advantages, response_token_ids)]

            all_query_token_ids.extend([sample["input_ids"]] * self.args.gens_per_sample)
            all_responses_token_ids.extend(response_token_ids)
            all_advantages.extend(per_token_advantages)

            stats["rewards"].extend(rewards)
            stats["non_stop_rate"].extend([fr != "stop" for fr in finish_reasons])
            stats["response_lengths"].extend([len(ids) for ids in response_token_ids])
            for rm in reward_metrics:
                for k, v in rm.items():
                    stats.setdefault(f"reward_metrics/{k}", []).append(v)

        episodes = {
            "all_query_token_ids": all_query_token_ids,
            "all_response_token_ids": all_responses_token_ids,
            "all_advantages": all_advantages,
        }

        return episodes, stats
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        total_response_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the GRPO policy gradient loss with KL penalty.
        """
        input_ids = batch["input_ids"]  # [batch_size, seq_len]
        attention_mask = batch["attention_mask"]  # [batch_size, seq_len]
        labels = batch["labels"]  # [batch_size, seq_len]
        labels_mask = batch["labels_mask"]  # [batch_size, seq_len]
        advantages = batch["advantages"]  # [batch_size, seq_len]

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "labels_mask": labels_mask,
        }

        logps = GRPOAlgorithm.compute_token_log_probs(self.policy_model, model_inputs, self.args.rollout_temp)  # [batch_size, seq_len-1]
        labels_mask = labels_mask[..., 1:].to(logps.dtype)  # [batch_size, seq_len-1]

        with torch.no_grad():
            entropy = (-(logps) * labels_mask).sum() / labels_mask.sum().clamp_min(1.0)  # scalar
            zero_advantages = close_to_zero(advantages[..., 1:], labels_mask)  # scalar

        policy_loss = -logps * advantages[..., 1:]  # [batch_size, seq_len-1]
        policy_loss = policy_loss * labels_mask  # [batch_size, seq_len-1]

        if self.args.kl_coeff > 0.0 and self.reference_model is not None:

            ref_logps = batch["ref_log_probs"]  # [batch_size, seq_len-1]
            
            ref_logratio = ref_logps - logps
            kl_penalty = torch.exp(ref_logratio) - 1 - ref_logratio  # [batch_size, seq_len-1]
            kl_penalty = kl_penalty * labels_mask  # [batch_size, seq_len-1]
            kl_penalty = self.args.kl_coeff * kl_penalty  # scalar
        else:
            kl_penalty = torch.zeros_like(policy_loss)

        loss = (policy_loss + kl_penalty).sum() / total_response_len  # scalar


        metrics = {
            "policy_loss": policy_loss.sum().item() / total_response_len.item(),
            "kl_penalty": kl_penalty.sum().item() / total_response_len.item(),
            "entropy": entropy.item() / total_response_len.item(),
            "entropy_total": entropy.item(),
            "zero_advantages": (zero_advantages / labels_mask.sum().clamp_min(1.0)).item(),
        }

        return loss, metrics

class GRPOTrainer:
    def __init__(
        self,
        task: CountdownTask,
        algorithm: GRPOAlgorithm,
        inference_engine: LLM,
        tokenizer: AutoTokenizer,
        args: argparse.Namespace,
        exp_dir: Path,
    ):
        self.task = task
        self.algorithm = algorithm
        self.inference_engine = inference_engine
        self.tokenizer = tokenizer
        self.args = args
        self.exp_dir = exp_dir
        
        self.sampler_rng = np.random.default_rng(seed=42)

    @staticmethod
    def prepare_model_inputs(
        query_token_ids: List[List[int]],
        response_token_ids: List[List[int]],
        advantages: List[List[float]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare padded model inputs with attention masks, labels, and advantages.
        """
        max_seq_len = max(len(q) + len(r) for q, r in zip(query_token_ids, response_token_ids))
        inputs = {"input_ids": [], "attention_mask": [], "labels": [], "advantages": [], "labels_mask": []}

        pad_token_id = 0  # Doesn't matter, will be masked
        ignore_index = -100

        for query, response, advantage in zip(query_token_ids, response_token_ids, advantages):
            combined_ids = query + response
            seq_len = len(combined_ids)

            # Create padded sequences
            input_ids = combined_ids + [pad_token_id] * (max_seq_len - seq_len)
            attention_mask = [1] * seq_len + [0] * (max_seq_len - seq_len)
            labels = [ignore_index] * len(query) + response + [ignore_index] * (max_seq_len - seq_len)
            labels_mask = [0] * len(query) + [1] * len(response) + [0] * (max_seq_len - seq_len)
            advantages_seq = [0.0] * len(query) + advantage + [0.0] * (max_seq_len - seq_len)

            assert len(input_ids) == max_seq_len
            assert len(attention_mask) == max_seq_len
            assert len(labels) == max_seq_len
            assert len(advantages_seq) == max_seq_len
            assert len(labels_mask) == max_seq_len

            inputs["input_ids"].append(input_ids)
            inputs["attention_mask"].append(attention_mask)
            inputs["labels"].append(labels)
            inputs["advantages"].append(advantages_seq)
            inputs["labels_mask"].append(labels_mask)

        # Convert to tensors
        return {
            k: torch.tensor(v, dtype=torch.long if k != "advantages" else torch.float, device=device)
            for k, v in inputs.items()
        }
        
    def dump_episodes(
        self,
        episodes: Dict[str, Any],
        episodes_stats: Dict[str, Any],
        step: int,
        is_eval: bool = False,
        do_save: bool = True,
    ) -> wandb.Table:
        query_token_ids = episodes["all_query_token_ids"]
        response_token_ids = episodes["all_response_token_ids"]
        rewards = episodes_stats["rewards"]
        response_lengths = episodes_stats["response_lengths"]

        query_texts = self.tokenizer.batch_decode(
            query_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        response_texts = self.tokenizer.batch_decode(
            response_token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        rank = process_rank()
        if (not is_eval) and is_main_process():
            print(f"########## Example 1 (Reward: {rewards[0]}, Response Length: {response_lengths[0]})")
            print(f"#### Query:\\n`{query_texts[0]}`")
            print(f"#### Response:\\n`{response_texts[0]}`\\n\\n")

            print(f"########## Example 2 (Reward: {rewards[1]}, Response Length: {response_lengths[1]})")
            print(f"#### Query:\\n`{query_texts[1]}`")
            print(f"#### Response:\\n`{response_texts[1]}`\\n\\n")

        if is_eval:
            episodes_dir = self.exp_dir / "eval_episodes"
        else:
            episodes_dir = self.exp_dir / "episodes"
        if dist.is_initialized():
            episodes_dir = episodes_dir / f"rank_{rank:02d}"
        episodes_dir.mkdir(parents=True, exist_ok=True)

        # Create wandb table
        table = wandb.Table(columns=["query", "response", "reward", "response_length"])
        for i in range(len(query_texts)):
            table.add_data(query_texts[i], response_texts[i], rewards[i], response_lengths[i])

        if not do_save:
            return table

        with open(episodes_dir / f"eps_{step:06d}.json", "w") as f:
            json.dump(
                [
                    {
                        "query": query_texts[i],
                        "response": response_texts[i],
                        "reward": rewards[i],
                    }
                    for i in range(len(query_texts))
                ],
                f,
            )

        return table

    def sample_eval(self, test_dataset: Dataset):
        if dist.get_rank() != 0:
            return None

        logger.info("Evaluating on eval set...")
        eval_sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=self.args.max_new_tokens,
            n=1,
            detokenize=False,
            stop_token_ids=[self.tokenizer.eos_token_id],
        )
        
        generations = self.inference_engine.generate(
            prompt_token_ids=test_dataset["input_ids"], sampling_params=eval_sampling_params
        )
        return generations

    def process_eval(self, step: int, test_dataset: Dataset, generations):
        if dist.get_rank() != 0:
            return None, None

        metrics = {
            "response_lengths": [],
            "rewards": [],
            "non_stop_rate": [],
        }

        all_query_token_ids = []
        all_responses_token_ids = []

        for i, sample in enumerate(test_dataset):
            query_token_ids = sample["input_ids"]
            response_token_ids = generations[i].outputs[0].token_ids
            finish_reason = generations[i].outputs[0].finish_reason

            response = self.tokenizer.decode(response_token_ids, skip_special_tokens=False)
            reward, reward_components = self.algorithm.reward_func(response, sample)

            all_query_token_ids.append(query_token_ids)
            all_responses_token_ids.append(response_token_ids)

            metrics["rewards"].append(reward)
            metrics["non_stop_rate"].append(finish_reason != "stop")
            metrics["response_lengths"].append(len(response_token_ids))
            for k, v in reward_components.items():
                metrics.setdefault(f"reward_metrics/{k}", []).append(v)

        episodes = {
            "all_query_token_ids": all_query_token_ids,
            "all_response_token_ids": all_responses_token_ids,
        }
        
        eval_stats = metrics
        
        eval_episode_table = self.dump_episodes(
            episodes=episodes,
            episodes_stats=eval_stats,
            step=step,
            is_eval=True,
        )
        return eval_stats, eval_episode_table

    def sample_rollouts(self, train_dataset: Dataset):
        # Sample training batch
        num_samples_per_step = self.args.eps_per_step // dist.get_world_size() // self.args.gens_per_sample
        indices = self.sampler_rng.choice(len(train_dataset), size=num_samples_per_step, replace=False)
        samples = train_dataset.select(indices)

        gen_time = time.time()

        # Sample responses
        outputs = self.inference_engine.generate(
            prompt_token_ids=samples["input_ids"],
            sampling_params=SamplingParams(
                n=self.args.gens_per_sample,
                temperature=self.args.rollout_temp,
                top_p=0.999, # TODO: Configurable?
                top_k=-1,
                max_tokens=self.args.max_new_tokens,
                detokenize=False,
                stop_token_ids=[self.tokenizer.eos_token_id],
            ),
        )
        all_generations = [list(g.token_ids) for out in outputs for g in out.outputs]
        all_finish_reasons = [g.finish_reason for out in outputs for g in out.outputs]

        logger.info(f"Generated {len(all_generations)} responses")
        logger.info(f"Time taken to generate {len(all_generations)} responses: {time.time() - gen_time} seconds")
        
        return samples, all_generations, all_finish_reasons

    def process_rollouts(self, step: int, samples, all_generations, all_finish_reasons):
        # Process responses and calculate rewards (GRPO)
        episodes, episodes_stats = self.algorithm.process_episodes(
            samples=samples,
            all_generations=all_generations,
            all_finish_reasons=all_finish_reasons,
        )

        self.inference_engine.sleep(1)
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(1)

        metrics = {}
        for k, v in episodes_stats.items():
            metrics.setdefault(k, []).extend(v)

        episode_table = self.dump_episodes(
            episodes=episodes,
            episodes_stats=episodes_stats,
            step=step,
            do_save=step % 10 == 0 or step == 0,
        )
        return episodes, metrics, episode_table

    def prepare_training_batch(self, episodes, device):
        # Prepare training batch
        model_inputs = self.prepare_model_inputs(
            query_token_ids=episodes["all_query_token_ids"],
            response_token_ids=episodes["all_response_token_ids"],
            advantages=episodes["all_advantages"],
            device=device,
        )

        if self.algorithm.reference_model is not None:
            logger.info("Moving reference model to GPU")
            self.algorithm.reference_model.module.to(device)
            self.algorithm.reference_model.eval()

            with torch.no_grad():
                ref_log_probs = []
                episodes_per_step_per_rank = self.args.eps_per_step // dist.get_world_size()
                for i in trange(
                    0,
                    episodes_per_step_per_rank,
                    self.args.batch_size_per_device,
                    desc="Computing reference logprobs",
                    disable=dist.get_rank() != 0,
                ):
                    batch = {k: v[i : i + self.args.batch_size_per_device] for k, v in model_inputs.items()}
                    ref_log_probs.append(
                        self.algorithm.compute_token_log_probs(
                            model=self.algorithm.reference_model,
                            inputs=batch,
                            temperature=self.args.rollout_temp,
                        )
                    )
                ref_log_probs = torch.cat(ref_log_probs)
                model_inputs["ref_log_probs"] = ref_log_probs
                del ref_log_probs

            # Free memory taken by reference model
            logger.info("Moving reference model back to CPU")
            self.algorithm.reference_model.cpu()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1)
            
        return model_inputs

    def train_on_batch(self, model_inputs, metrics):
        # Calculate losses and update model
        self.algorithm.policy_model.train()
        total_response_len = (model_inputs["labels"] != -100).sum()
        train_time = time.time()

        episodes_per_step_per_rank = self.args.eps_per_step // dist.get_world_size()
        for i in trange(
            0,
            episodes_per_step_per_rank,
            self.args.batch_size_per_device,
            desc="Gradient Accumulation",
            disable=dist.get_rank() != 0,
        ):
            batch = {k: v[i : i + self.args.batch_size_per_device] for k, v in model_inputs.items()}

            # Compute policy gradient loss (GRPO)
            loss, loss_metrics = self.algorithm.compute_loss(
                batch=batch,
                total_response_len=total_response_len,
            )

            # Track metrics
            metrics.setdefault("loss", []).append(loss.item())
            grad_norm = self.algorithm.policy_model.get_global_grad_norm()
            if grad_norm is not None:
                grad_norm = grad_norm.item()
            metrics.setdefault("grad_norm", []).append(grad_norm)
            for k, v in loss_metrics.items():
                metrics.setdefault(k, []).append(v.item() if isinstance(v, torch.Tensor) else v)

            # Backpropagation and optimization step
            self.algorithm.policy_model.backward(loss)
            del loss, loss_metrics

            self.algorithm.policy_model.step()

        logger.info(f"Time taken to train: {time.time() - train_time} seconds")

    def train(self, train_dataset: Dataset, test_dataset: Dataset, begin_iter: int = 0):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        
        # Frankly, I also have no idea why this is here.
        # Educated guess is to restore sampler state in the case
        # of loading from a checkpoint.
        if begin_iter > 0:
            logger.info(f"Advancing trainer sampler by {begin_iter} steps")
            num_samples_per_step = self.args.eps_per_step // (dist.get_world_size() * self.args.gens_per_sample)
            for _ in range(begin_iter):
                 _ = self.sampler_rng.choice(
                        len(train_dataset), 
                        size=num_samples_per_step, 
                        replace=False
                    )
        
        # Main training loop
        for step in trange(begin_iter, self.args.n_steps):
            logger.info(f"Step {step}/{self.args.n_steps}")

            #########################################################
            # Evaluation
            #########################################################

            eval_stats = None
            if step % 25 == 0 and step > 0:
                if is_main_process():
                    eval_generations = self.sample_eval(test_dataset)
                    eval_stats, eval_table = self.process_eval(step, test_dataset, eval_generations)
                    wandb.log({"eval/episodes": eval_table, "step": step})
                dist.barrier(device_ids = [device.index])

            #########################################################
            # Rollout Sampling
            #########################################################

            samples, all_generations, all_finish_reasons = self.sample_rollouts(train_dataset)

            #########################################################
            # Process Rollouts
            #########################################################
            episodes, metrics, episode_table = self.process_rollouts(
                step, samples, all_generations, all_finish_reasons
            )

            #########################################################
            # Prepare Training Batch for Backpropagation
            #########################################################
            model_inputs = self.prepare_training_batch(episodes, device)

            #########################################################
            # Train on Batch
            #########################################################
            self.train_on_batch(model_inputs, metrics)

            #########################################################
            # Update vLLM with new model weights
            #########################################################

            self.inference_engine.wake_up()
            load_model_into_vllm(self.algorithm.policy_model, self.inference_engine)

            #########################################################
            # Log metrics
            #########################################################

            if is_main_process():
                train_metrics = {k: np.mean(v) for k, v in metrics.items() if None not in v}
                train_metrics["learning_rate"] = self.algorithm.policy_model.get_lr()[0]
                logs = {
                    "step": step,
                    f"episodes/step_{step:06d}": episode_table,
                    **{f"train/{k}": v for k, v in train_metrics.items()},
                }
                if eval_stats is not None:
                    logs.update({f"eval/{k}": np.mean(v) for k, v in eval_stats.items()})
                wandb.log(logs)

                selected_keys = [
                    "train/kl_penalty",
                    "train/rewards",
                    "train/reward_metrics/format_reward",
                    "train/reward_metrics/verification_reward",
                    "train/response_lengths",
                    "eval/rewards",
                    "eval/reward_metrics/format_reward",
                    "eval/reward_metrics/verification_reward",
                ]
                selected_metrics = {k: float(logs[k]) for k in selected_keys if k in logs}
                logger.info(f"KEY METRICS: {selected_metrics}")

            if step % 50 == 0 and step != 0:
                ckpt_dir = self.exp_dir / "checkpoints" / f"ckpt_{step:06d}"

                logger.info("Saving HF model")
                if is_main_process():
                    self.algorithm.policy_model.module.save_pretrained(
                        str(ckpt_dir / "hf_model")
                    )
                    self.tokenizer.save_pretrained(
                        str(ckpt_dir / "hf_model")
                    )
                dist.barrier(device_ids=[device.index])

                # WARNING: DeepSpeed checkpoints can be ENORMOUS 
                #          (24GB for a 1.5B model) due to storing other things 
                #          like optimizer states, activations, etc.
                #          If you do not have enough space,
                #          consider commenting this part out.
                logger.info("Saving DeepSpeed checkpoint")
                self.algorithm.policy_model.save_checkpoint(
                    str(ckpt_dir / "deepspeed")
                )

                if is_main_process():
                    clean_up_checkpoints(
                        exp_dir=self.exp_dir,
                        keep_every_n_steps=50,
                        exclude=[ckpt_dir],
                    )
                dist.barrier(device_ids = [device.index])

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def find_free_port():
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


# def evaluate_on_test_set(
#     inference_engine: LLM,
#     test_dataset: Dataset,
#     tokenizer: AutoTokenizer,
#     eos_token: str,
#     eval_sampling_params: SamplingParams,
#     reward_func: Callable[[str, Dict[str, Any]], Tuple[float, Dict[str, float]]],
# ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
#     """
#     Evaluate the model on a test dataset by generating responses and computing rewards.
#     """
#     generations = inference_engine.generate(
#         prompt_token_ids=test_dataset["input_ids"], sampling_params=eval_sampling_params
#     )

#     metrics = {
#         "response_lengths": [],
#         "rewards": [],
#         "non_stop_rate": [],
#     }

#     all_query_token_ids = []
#     all_responses_token_ids = []

#     for i, sample in enumerate(test_dataset):
#         query_token_ids = sample["input_ids"]
#         response_token_ids = generations[i].outputs[0].token_ids
#         finish_reason = generations[i].outputs[0].finish_reason

#         response = tokenizer.decode(response_token_ids, skip_special_tokens=False)
#         reward, reward_components = reward_func(response, sample)

#         all_query_token_ids.append(query_token_ids)
#         all_responses_token_ids.append(response_token_ids)

#         metrics["rewards"].append(reward)
#         metrics["non_stop_rate"].append(finish_reason != "stop")
#         metrics["response_lengths"].append(len(response_token_ids))
#         for k, v in reward_components.items():
#             metrics.setdefault(f"reward_metrics/{k}", []).append(v)

#     episodes = {
#         "all_query_token_ids": all_query_token_ids,
#         "all_response_token_ids": all_responses_token_ids,
#     }

#     return episodes, metrics


def find_last_checkpoint(exp_dir: Path) -> Tuple[Optional[Path], Optional[int]]:
    checkpoint_dir = exp_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None, None
    checkpoints = list(checkpoint_dir.glob("ckpt_*"))
    # Filter out directories that don't have a deepspeed subdirectory
    checkpoints = [ckpt for ckpt in checkpoints if (ckpt / "deepspeed").exists()]
    if not checkpoints:
        return None, None
    ckpt_path = max(checkpoints, key=lambda x: int(x.stem.split("_")[-1]))
    ckpt_iter = int(ckpt_path.stem.split("_")[-1])
    return ckpt_path, ckpt_iter


def load_model_into_vllm(model: Union[DeepSpeedEngine, PreTrainedModel], llm: LLM) -> None:
    """
    Load weights from a HuggingFace model (either wrapped in DeepSpeed or not) into a vLLM inference engine.
    """
    state_dict = model.module.state_dict() if isinstance(model, DeepSpeedEngine) else model.state_dict()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(state_dict.items())


def initialize_training_process_group(rank: int, world_size: int, device_id: Optional[int] = None):
    """
    Initialize the PyTorch distributed process group for multi-GPU training using NCCL backend.
    """
    master_addr = "localhost"
    master_training_port = 8237

    os.environ["LOCAL_RANK"] = str(rank)

    if device_id is None:
        device_id = rank
    
    torch.cuda.set_device(device_id)

    if is_main_process():
        print(
            f"{'#' * 80}\n" f"# Initializing the training NCCL PG with\n" f"# world_size={world_size} \n" f"{'#' * 80}"
        )

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_training_port}",
        world_size=world_size,
        rank=rank,
        timeout=timedelta(hours=1),
    )
    dist.barrier(device_ids=[device_id])
    print(
        f"Rank{rank}: training NCCL PG initialized. "
        f"(world_size={world_size}, local_rank={rank}, gpu_id={torch.cuda.current_device()})"
    )

def clean_up_checkpoints(
    exp_dir: Path, keep_every_n_steps: Optional[int] = None, exclude: Optional[List[Path]] = None
) -> None:
    """
    Clean up checkpoint directories by removing unnecessary files and directories.
    """
    if exclude is None:
        exclude = []

    checkpoint_dir = exp_dir / "checkpoints"
    for ckpt in checkpoint_dir.glob("ckpt_*"):
        if keep_every_n_steps is None or ckpt in exclude:
            continue

        ckpt_iter = int(ckpt.stem.split("_")[-1])
        if ckpt_iter % keep_every_n_steps == 0:
            # Remove non-hf_model files and dirs
            removed_files_and_dirs = []
            for file in ckpt.iterdir():
                if file.name not in ["hf_model"]:
                    try:
                        removed_files_and_dirs.append(file.name)
                        if file.is_dir():
                            shutil.rmtree(file, ignore_errors=True)
                    except Exception as e:
                        print(f"Error removing {file}: {e}")
            if len(removed_files_and_dirs) > 0:
                print(f"Removed non-hf_model files and dirs: of checkpoint {ckpt.name}")

            continue

        print(f"Removing checkpoint {ckpt}")
        shutil.rmtree(ckpt)

ZERO_EPS = 1e-8
def close_to_zero(tensor: torch.Tensor, mask: torch.Tensor, threshold: float = ZERO_EPS) -> torch.Tensor:
    """
    Computes the number of values in the tensor that are close to zero.
    """
    close_to_zero_mask = torch.abs(tensor) < threshold
    num_close_to_zero = (close_to_zero_mask * mask).sum()
    return num_close_to_zero


# -----------------------------------------------------------------------------
# Main Training Loop
# -----------------------------------------------------------------------------

def main(rank: int, args: argparse.Namespace):
    
    # Check visible devices
    if not len(args.visible_devices):
        raise ValueError("No visible devices specified. Set CUDA_VISIBLE_DEVICES environment variable.")
    if rank >= len(args.visible_devices):
        raise ValueError(f"Rank {rank} exceeds the number of visible devices: {len(args.visible_devices)}")
    
    # Initialize process group and set device
    initialize_training_process_group(
        rank = rank, 
        world_size = args.n_procs, 
        device_id = args.visible_devices[rank]
    )
    curr_cuda_device = torch.device(f"cuda:{args.visible_devices[rank]}")
    logging.info(f"Current CUDA device set to: {curr_cuda_device}")

    # Safety measure: change rank.
    os.environ["LOCAL_RANK"] = str(args.visible_devices[rank])

    # Optional: Mute non-main processes' logging
    # However, it's best to be able to keep tabs on all processes for debugging.
    # if not is_main_process():
    #     logger.setLevel(logging.ERROR)

    # Start debug mode (I have no idea what this actually does.)
    if args.debug and args.n_procs == 1:
        import debugpy

        debugpy.listen(5678)
        logger.info("Waiting for debugger to attach...")
        debugpy.wait_for_client()
        logger.info("Debugger attached")

    ############################################
    # Hyperparameters
    ############################################

    # # Model configuration
    # MODEL_NAME = args.model_name

    # # RL parameters
    # # Total number of training steps
    # n_steps = args.n_steps
    # # Number of episodes to collect per step for training
    # EPISODES_PER_STEP = args.eps_per_step
    # EPISODES_PER_STEP_PER_RANK = EPISODES_PER_STEP // dist.get_world_size()
    # # Number of responses to generate for each input prompt
    # GENERATIONS_PER_SAMPLE = args.gens_per_sample
    # # Controls how much the policy can deviate from the reference model
    # KL_COEFFICIENT = args.kl_coeff

    # # Training hyperparameters
    # # Batch size for each GPU device during training
    # batch_size_per_device = args.batch_size_per_device
    # # Learning rate for model updates
    # LEARNING_RATE = args.learning_rate

    # # Sampling parameters
    # # Maximum number of tokens to generate in each response
    # MAX_RESPONSE_TOKENS = args.max_response_tokens
    # # Controls randomness in generation (higher = more random)
    # TEMPERATURE = args.temperature
    # # Nucleus sampling parameter (1.0 = disabled)
    # TOP_P = 0.999  # to avoid sampling unused tokens absent from tokenizer
    # # Top-k sampling parameter (-1 = disabled)
    # TOP_K = -1  # no top k
    
    # DeepSpeed configuration
    # Ensure batch size divisibility
    if args.eps_per_step % world_size() != 0:
        raise ValueError(
            f"EPISODES_PER_STEP ({args.eps_per_step}) "
            f"must be divisible by world size ({world_size()})."
        )
    
    eps_per_step_rank = args.eps_per_step // world_size()
    
    if eps_per_step_rank % args.batch_size_per_device != 0:
        raise ValueError(
            f"eps_per_step_rank ({eps_per_step_rank}) must be divisible by "
            f"batch_size_per_device ({args.batch_size_per_device})."
        )

    # Use DeepSpeed to perform backpropagation and optimization
    deepspeed_config = {
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2, "overlap_comm": False},
        "train_batch_size": args.eps_per_step,
        "train_micro_batch_size_per_gpu": args.batch_size_per_device,
        # Reduces memory usage on each GPU by backpropagating
        # only after accumulating gradients over multiple mini-batches.
        "gradient_accumulation_steps": eps_per_step_rank // args.batch_size_per_device,
        "gradient_clipping": args.gradient_clipping,
        "optimizer": {
            "type": "AdamW", # In case you want weight decay
            "params": {
                "lr": args.learning_rate,
                # Lowering 2nd order momentum for stability.
                "betas": (0.9, 0.95), 
                "eps": 1e-8,
                "weight_decay": args.weight_decay,
                "torch_adam": True,
                "fused": True,
            },
        },
    }
    
    if args.use_ref_based_rlvr:
        ref_deepspeed_config = {
            "bf16": {"enabled": True},
            "train_batch_size": args.eps_per_step,
            "train_micro_batch_size_per_gpu": args.batch_size_per_device,
            "gradient_accumulation_steps": eps_per_step_rank // args.batch_size_per_device,
        }

    dist_barrier(device_ids = [torch.cuda.current_device()])

    # Ready logging and experiment directory
    model_name_short = args.model_name.split("/")[-1]
    if args.run_id is None:
        RUN_NAME = f"{model_name_short}_temp{args.rollout_temp}_kl{args.kl_coeff}_lr{args.lr}_grpo"
    else:
        RUN_NAME = args.run_id
    EXP_DIR = Path.home() / "scratch" / "nano_aha_moment" / RUN_NAME
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Logs and Checkpoints will be saved to: {EXP_DIR}")

    ############################################
    # Prompts and Dataset
    ############################################

    # It is possible to share HuggingFace FastTokenizers across processes.
    # For backwards compatibility AND simplicity,
    # we will just load a tokenizer per process.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    EOS_TOKEN_ID = tokenizer.eos_token_id
    EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)

    # P
    train_dataset, test_dataset = CountdownTask.load_and_preprocess_dataset(
        dataset_name = args.dataset_name,
        tokenizer = tokenizer,
    )
    
    orig_train_dataset_size = len(train_dataset) * dist.get_world_size() # Approximate original size

    # Shard the training dataset
    train_dataset = train_dataset.shard(num_shards=dist.get_world_size(), index=dist.get_rank())

    logger.info(f"Train dataset size: {orig_train_dataset_size}; each rank will process {len(train_dataset)} samples")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    ############################################
    # Initialize Models
    ############################################

    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=torch.cuda.current_device(),
    )
    if args.use_ref_based_rlvr:
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map=torch.cuda.current_device(),
        )
    else:
        reference_model = None
    policy_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    see_memory_usage(
        "Before initializing DeepSpeed engines", 
        force = is_main_process()
    )

    # Initialize DeepSpeed engines
    policy_model, *_ = deepspeed.initialize(
        model=policy_model,
        config=deepspeed_config,
        model_parameters=policy_model.parameters(),
    )
    reference_model, *_ = deepspeed.initialize(
        model=reference_model,
        config=ref_deepspeed_config,
    )

    if reference_model is not None:
        reference_model.module.cpu()
    dist.barrier(device_ids=[torch.cuda.current_device()])

    ############################################
    # Initialize vLLM engine for inference
    # This will power our rollouts
    ############################################

    see_memory_usage(
        "Before initializing inference engine", 
        force = is_main_process()
    )

    if not is_main_process():
        # Disable root vllm logger for non-main ranks
        vllm_logger = logging.getLogger("vllm")
        vllm_logger.setLevel(logging.ERROR)

    inference_engine = LLM(
        model = args.model_name,
        skip_tokenizer_init = False,
        gpu_memory_utilization = args.vllm_gpu_memory_utilization,
        enable_prefix_caching = not args.vllm_disable_prefix_caching,
        swap_space = 4,
        scheduling_policy = "fcfs",
        dtype = torch.bfloat16,
        # Make space for system prompt + input prompt
        max_model_len = args.max_model_len,
        enable_sleep_mode = True,
        device = f"cuda:{torch.cuda.current_device()}",
        tensor_parallel_size = 1,
    )

    see_memory_usage(
        "After initializing inference engine",
        force = is_main_process()
    )

    # Wandb for logging.
    if is_main_process():
        wandb.init(
            project="nano-aha-moment",
            name=RUN_NAME,
            resume="allow",
            config={
                "model_name": args.model_name,
                "learning_rate": args.lr,
                "n_steps": args.n_steps,
                "episodes_per_step": args.eps_per_step,
                "rollouts_per_episode": args.gens_per_sample,
                "kl_coefficient": args.kl_coeff,
                "temperature": args.rollout_temp,
                "algorithm": "grpo",
            },
        )

    sampler_rng = np.random.default_rng(seed = 42)
    NUM_SAMPLES_PER_STEP = eps_per_step_rank // args.gens_per_sample

    # Load checkpoint if it exists
    # Good for training on unreliable machines such as cloud compute instances.
    begin_iter = 0
    ckpt_path, ckpt_iter = find_last_checkpoint(EXP_DIR)
    if ckpt_path is not None:
        logger.info(f"Resuming from checkpoint {ckpt_path} at step {ckpt_iter}")
        out = policy_model.load_checkpoint(ckpt_path / "deepspeed")
        if out is None:
            raise RuntimeError(f"Failed to load checkpoint {ckpt_path}")
        begin_iter = ckpt_iter + 1
        load_model_into_vllm(policy_model, inference_engine)

        logger.info(f"Skipping {ckpt_iter} rounds of samples")
        for _ in trange(ckpt_iter, disable = not is_main_process()):
            _ = sampler_rng.choice(len(train_dataset), size=NUM_SAMPLES_PER_STEP, replace=False)

    # Initialize Algorithm and Trainer
    task = CountdownTask()
    algorithm = GRPOAlgorithm(
        policy_model = policy_model,
        reference_model = reference_model,
        tokenizer = tokenizer,
        reward_func = task.get_reward,
        args = args,
    )
    trainer = GRPOTrainer(
        task = task,
        algorithm = algorithm,
        inference_engine = inference_engine,
        tokenizer = tokenizer,
        args = args,
        exp_dir = EXP_DIR,
    )
    
    ############################################
    # And finally, the GRPO training loop
    ############################################
    trainer.train(train_dataset, test_dataset, begin_iter)

    dist.destroy_process_group()


# -----------------------------------------------------------------------------
# Program Entry Point
# The only important thing here are the command-line arguments,
# which you can equally access with `python true_one_script.py --help`
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Argument Parser
    arg_parser = argparse.ArgumentParser(
        description = "Train R1 model with GRPO"
    )
    arg_parser.add_argument(
        "--kl_coeff", 
        type = float, default = 0.0, 
        help = "KL coefficient for GRPO"
    )
    arg_parser.add_argument(
        "--rollout_temp", 
        type = float, default = 1.0, 
        help = "Temperature for rollouts (sampling stage)"
    )
    arg_parser.add_argument(
        "--eval_temp",
        type = float,
        default = 0.6,
        help = "Temperature for rollouts (evaluation stage)",
    )
    arg_parser.add_argument(
        "--model_name", 
        type = str, default = "Qwen/Qwen2.5-3B", 
        help = "Model name/path"
    )
    arg_parser.add_argument(
        "--dataset_name", 
        type = str, default = "Jiayi-Pan/Countdown-Tasks-3to4", 
        help = "HuggingFace dataset name"
    )
    arg_parser.add_argument(
        "--batch_size_per_device", 
        type = int, default = 8, 
        help = "Batch size per device"
    )
    arg_parser.add_argument(
        "--max_new_tokens", 
        type = int, default = 1024, 
        help = "Max new tokens to generate per response"
    )
    arg_parser.add_argument(
        "--lr", 
        type = float, default = 1e-6, 
        help = "Learning rate for training"
    )
    arg_parser.add_argument(
        "--debug", 
        action = "store_true", 
        help = "Activate debug mode"
    )
    arg_parser.add_argument(
        "--run_id", 
        type = str, default = None, 
        help = "Run ID"
    )
    arg_parser.add_argument(
        "--n_procs", 
        type = int, default = 1, 
        help = "Number of processes (data parallelism) to use"
    )
    arg_parser.add_argument(
        "--n_steps", 
        type = int, default = 1000, 
        help = "Total number of training steps"
    )
    arg_parser.add_argument(
        "--eps_per_step", 
        type = int, default = 64, 
        help = "Number of episodes to collect per step"
    )
    arg_parser.add_argument(
        "--gens_per_sample", 
        type = int, default = 8, 
        help = "More gens_per_sample -> Richer advantage estimates"
    )
    arg_parser.add_argument(
        "--visible_devices",
        type = str,
        default = "0",
        help = "Comma-separated list of visible devices. E.g., '0,1,2,3'",
    )
    
    # vLLM parameters
    arg_parser.add_argument(
        "--vllm_gpu_memory_utilization", 
        type = float, default = 0.3, 
        help = "vLLM GPU memory utilization"
    )
    arg_parser.add_argument(
        "--vllm_disable_prefix_caching", 
        action = "store_true", 
        help = "Disable vLLM prefix caching"
    )
    arg_parser.add_argument(
        "--max_model_len", 
        type = int, default = 2048, 
        help = "Max model length (context window size) for vLLM"
    )

    # DeepSpeed parameters
    arg_parser.add_argument(
        "--gradient_clipping", 
        type = float, default = 1.0, 
        help = "Gradient clipping value"
    )
    arg_parser.add_argument(
        "--weight_decay", 
        type = float, default = 0.0, 
        help = "Weight decay"
    )

    def parse_args(args = None) -> argparse.Namespace:

        # Technically you can continue on to do some validation here,
        # but I trust that people at least know what the assumptions
        # on the hyperparameters are....?
        parsed_args = arg_parser.parse_args(args)

        # Process visible devices into a list of integers
        parsed_args.visible_devices = list(
            map(
                int,
                parsed_args.visible_devices.split(",")
            )
        )
        parsed_args.use_ref_based_rlvr = parsed_args.kl_coeff > ZERO_EPS
        return parsed_args

    script_args = parse_args()

    n_gpus = torch.cuda.device_count()

    if script_args.n_procs > n_gpus:
        raise ValueError(
            f"Requested {script_args.n_procs} processes, "
            f"but only {n_gpus} GPUs are available."
        )

    if script_args.n_procs == 1:
        main(rank = 0, args = script_args)
    else:
        # Common footgun: args is not picklable.
        # Make sure you are not passing any class objects in args!
        torch.multiprocessing.spawn(
            main, 
            args = (script_args, )
        )
