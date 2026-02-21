import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

class DTRModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", model_name=None):
        self.device = device
        self.default_system_prompt = "You are a helpful mathematical reasoning assistant."
        
        # We switched to the original full Hugging Face repo because `transformers` 
        # doesn't yet support the `lfm2` architecture for GGUF files.
        # This will download the bfloat16/float16 weights, which easily fits in VRAM for a 1.2B model.
        original_repo = "DavidAU/LFM2.5-1.2B-Thinking-Claude-4.6-Opus-Heretic-Uncensored-DISTILL"
        self.model_id = (
            model_name
            or os.environ.get("DTR_MODEL_ID")
            or original_repo
        )
        
        adapter_base_model_id = None
        if self._is_peft_adapter(self.model_id):
            peft_cfg = PeftConfig.from_pretrained(self.model_id)
            adapter_base_model_id = peft_cfg.base_model_name_or_path

        tokenizer_source = adapter_base_model_id or self.model_id
        print(f"Loading tokenizer from {tokenizer_source}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
        
        print(f"Loading model from {self.model_id}...")

        is_cuda = str(self.device).startswith("cuda")
        if is_cuda and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        elif is_cuda:
            dtype = torch.float16
        else:
            dtype = torch.float32

        model_kwargs = {
            "cache_dir": "./models",
            "low_cpu_mem_usage": True,
            "torch_dtype": dtype,
            "output_hidden_states": True,
            "trust_remote_code": True,
        }
        if is_cuda:
            # Respect requested CUDA device instead of hard-coding cuda:0.
            model_kwargs["device_map"] = {"": self.device}

        # Support loading either a full base model or a LoRA adapter directory.
        if adapter_base_model_id is not None:
            base_model_id = adapter_base_model_id
            print(f"Detected PEFT adapter. Loading base model: {base_model_id}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                **model_kwargs
            )
            self.model = PeftModel.from_pretrained(base_model, self.model_id)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            )
        if not is_cuda:
            self.model.to(self.device)
        
        # Unembedding matrix W_U and final normalization layer.
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        self.lm_head = self._resolve_lm_head(base)
        self.final_norm = self._resolve_final_norm(base)

    @staticmethod
    def _is_peft_adapter(model_id):
        if os.path.isdir(model_id):
            return os.path.exists(os.path.join(model_id, "adapter_config.json"))
        return False

    @staticmethod
    def _resolve_lm_head(model):
        if hasattr(model, "lm_head"):
            return model.lm_head
        raise AttributeError("Could not find lm_head on loaded model.")

    @staticmethod
    def _resolve_final_norm(model):
        candidate_paths = [
            ("model", "embedding_norm"),
            ("model", "norm"),
            ("norm",),
            ("transformer", "ln_f"),
        ]
        for path in candidate_paths:
            current = model
            ok = True
            for attr in path:
                if hasattr(current, attr):
                    current = getattr(current, attr)
                else:
                    ok = False
                    break
            if ok:
                return current
        raise AttributeError("Could not resolve final normalization layer for this model.")
        
    def _format_prompt(self, prompt, system_prompt=None):
        # Format prompt using the model's chat template.
        active_system_prompt = system_prompt or self.default_system_prompt
        messages = [
            {"role": "system", "content": active_system_prompt},
            {"role": "user", "content": prompt}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def prepare_inputs(self, prompt, system_prompt=None):
        prompt_formatted = self._format_prompt(prompt, system_prompt=system_prompt)
        inputs = self.tokenizer(prompt_formatted, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        return inputs, prompt_length, prompt_formatted

    def generate_with_hidden_states(
        self,
        prompt,
        max_new_tokens=50,
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        system_prompt=None,
        return_prompt_metadata=False,
        output_hidden_states=True
    ):
        inputs, prompt_length, prompt_formatted = self.prepare_inputs(
            prompt, system_prompt=system_prompt
        )
        outputs = self.generate_from_model_inputs(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            output_hidden_states=output_hidden_states,
        )

        if return_prompt_metadata:
            return outputs, prompt_length, prompt_formatted
        return outputs

    def generate_from_model_inputs(
        self,
        model_inputs,
        max_new_tokens=50,
        do_sample=False,
        temperature=None,
        top_k=None,
        top_p=None,
        output_hidden_states=True
    ):
        # Ensure config requests hidden states when needed.
        self.model.config.output_hidden_states = bool(output_hidden_states)

        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "return_dict_in_generate": True,
            "output_hidden_states": bool(output_hidden_states),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if top_p is not None:
                gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                **gen_kwargs
            )
        return outputs

    @staticmethod
    def extract_generated_hidden_states(outputs):
        """
        Convert HF generation hidden states into a per-generated-token list.
        For step 0 (prompt forward pass), only the last prompt token state is
        relevant for predicting the first generated token.
        """
        hidden_states_list = []
        if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
            return hidden_states_list

        for step_idx, step_hidden_states in enumerate(outputs.hidden_states):
            if step_idx == 0:
                extracted_layers = [layer[:, -1:, :] for layer in step_hidden_states]
                hidden_states_list.append(extracted_layers)
            else:
                hidden_states_list.append(step_hidden_states)

        return hidden_states_list
