from pydantic import BaseModel, conlist, dataclasses
from typing import Literal
from typing import List, Optional
from textwrap import dedent
from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter
from lmformatenforcer import JsonSchemaParser
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler,
)
from exllamav2.generator.filters import (
    ExLlamaV2PrefixFilter
)
import time, json, sys

model_templates = {
    "dolphin-2.6-mixtral-8x7b-3.5bpw-h6-exl2": {
        'template': dedent(
            """<|im_start|>system
            {system_prompt}<|im_end|>
            <|im_start|>user
            {prompt}<|im_end|>
            <|im_start|>assistant
            {start_response}"""
        ),
        'eos_tag': '<|im_end|>'
    },
    "miqu-1-70b-sf-2.4bpw-h6-exl2": {
        'template': dedent(
            """[INST]system
            {system_prompt}[/INST]
            [INST]user
            {prompt}[/INST]
            [INST]assistant
            {start_response}"""
        ),
        'eos_tag': '[/INST]'
    }
}

class EXL2Model:

    global model_templates

    def __init__(self, model_name, model_dir="models", seq_len=16384, verbose=True, enable_stats=True):
        self.verbose = verbose
        self.enable_stats = enable_stats
        print("Loading model: " + model_name)
        self.model_name = model_name
        self.model_dir = model_dir + '/' if model_dir[-1] != '/' else model_dir
        self.seq_len = seq_len
        self.config = ExLlamaV2Config(self.model_dir + self.model_name)
        self.config.prepare()
        self.config.max_seq_len = self.seq_len
        self.model = ExLlamaV2(self.config)
        self.cache = ExLlamaV2Cache_Q4(self.model, lazy = True)
        self.model.load_autosplit(self.cache)
        self.tokenizer = ExLlamaV2Tokenizer(self.config)
        self.generator = ExLlamaV2StreamingGenerator(self.model, self.cache, self.tokenizer)
        self.generator.speculative_ngram = True
        self.generator.warmup()
        print("Model loaded.")


    def generate(
                self, 
                prompt,
                system_prompt="You are a helpful assistant.",
                response_prompt="",
                guidance=None, 
                max_new_tokens = 2000, 
                eos_bias = False,
                repeat_penalty = 1.0
            ):
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 0.4
        settings.top_k = 0
        settings.top_p = 0.5
        settings.token_repetition_penalty = repeat_penalty
        
        filters = []
        if guidance is not None:
            filters = self._processPydanticObject(guidance)
            settings.filters = filters
        settings.filter_prefer_eos = eos_bias
        input_text = model_templates[self.model_name]['template'].format(
                prompt=prompt, 
                system_prompt="You are a helpful assistant.",
                start_response=resp_start
        )
        input_ids = self.tokenizer.encode(input_text)
        prompt_tokens = input_ids.shape[-1]
        time_begin_prompt = time.time()
        self.generator.set_stop_conditions([self.tokenizer.eos_token_id])
        self.generator.begin_stream_ex(input_ids, settings)
        time_begin_stream = time.time()
        generated_tokens = 0
        
        if self.verbose:
            print("--------------------------------------------------")
            print(prompt)
            print(" ------>" + (" (filtered)" if len(filters) > 0 else ""))
        
        result = ""
        if self.verbose:
            while True:
                res = self.generator.stream_ex()
                result += res["chunk"] 
                generated_tokens += 1
                print(res["chunk"], end = "")
                sys.stdout.flush()
                if res["eos"] or generated_tokens == max_new_tokens: break
        else:
            while True:
                res = self.generator.stream_ex()
                result += res["chunk"]
                generated_tokens += 1
                if res["eos"] or generated_tokens == max_new_tokens: break

        result = result.replace(model_templates[self.model_name]['eos_tag'], "")

        time_end = time.time()
        time_prompt = time_begin_stream - time_begin_prompt
        time_tokens = time_end - time_begin_stream

        if guidance is not None:
            result = json.loads(result)
            result = json.dumps(result, indent=4)
            if self.verbose:
                print("\n")
                print("Parsed JSON:" , result)

        if self.enable_stats:
            print("\n")
            print(f"Prompt processed in {time_prompt:.2f} seconds, {prompt_tokens} tokens, {prompt_tokens / time_prompt:.2f} tokens/second")
            print(f"Response generated in {time_tokens:.2f} seconds, {generated_tokens} tokens, {generated_tokens / time_tokens:.2f} tokens/second")
            print("\n")        
        
        return result
    

    def _processPydanticObject(self, obj: BaseModel):
        schema_parser = JsonSchemaParser(obj.schema())
        lmfe_filter = ExLlamaV2TokenEnforcerFilter(schema_parser, self.tokenizer)
        prefix_filter = ExLlamaV2PrefixFilter(self.model, self.tokenizer, "{")
        return [lmfe_filter, prefix_filter]
    

# Test it out
if __name__ == "__main__":

    class Variable(BaseModel):
        name: str
        type: str
        description: str

    class Implementation(BaseModel):
        code: str
        test_code: str
        variables: List[Variable]

    class Step(BaseModel):
        title: str
        description: str
        implementation: Implementation
        alternative_approaches: conlist(Implementation, min_length=1, max_length=10)

    class Plan(BaseModel):
        name: str
        steps: List[Step]

    #exl2 = EXL2Model(model_name="dolphin-2.6-mixtral-8x7b-3.5bpw-h6-exl2", verbose=True)
    exl2 = EXL2Model(model_name="miqu-1-70b-sf-2.4bpw-h6-exl2", verbose=True)


    prompt = """Write a full implementation plan in Python to run sentiment analysis on a dataset. """
    resp_start = "Sure, "
    result = exl2.generate(prompt, eos_bias=False, max_new_tokens=4000, repeat_penalty=1.04, response_prompt=resp_start, guidance=Plan)
    print("\nRESPONSE ----\n{resp_start}", result)
