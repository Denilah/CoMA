import argparse
import pprint
import os
from dataclasses import dataclass, field
from transformers import GenerationConfig
from tqdm import tqdm
import torch
import gradio as gr
import transformers
from eval.human_eval.data import write_jsonl, read_problems, stream_jsonl
from train import ModelArguments, smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
    DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request. "
    "Write code that appropriately completes the request.\n\n"
    "### Instruction:\nCreate a Python script for this problem:\n{instruction}\n\n### Response:"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='$GEMINI_PRETRAIN', help="")
parser.add_argument('--output_path', type=str, help="")
parser.add_argument('--start_index', type=int, default=0, help="")
parser.add_argument('--end_index', type=int, default=164, help="")
parser.add_argument('--temperature', type=float, default=0.2, help="")
parser.add_argument('--N', type=int, default=200, help="")
parser.add_argument('--max_len', type=int, default=512, help="")
parser.add_argument('--decoding_style', type=str, default='sampling', help="")
parser.add_argument('--num_seqs_per_iter', type=int, default=50, help='')
parser.add_argument('--overwrite', action='store_true', help='')
parser.add_argument('--load_in_8bit', type=bool, default=False, help='')
parser.add_argument('--inference_dtype', type=str, default="float32", help="The dtype to use for inference.")
parser.add_argument('--launch_gradio', type=bool, default=False, help="Whether to use user input for prompting or a fixed eval list.")
args = parser.parse_args()

def initialize():
    global model, tokenizer

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_8bit=args.load_in_8bit,
        torch_dtype=torch.float16 if args.inference_dtype == "float16" else torch.float32,
    )
    if not args.load_in_8bit:
        model.to(device)
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        use_fast=False,
        model_max_length=args.max_len,
    )

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens(
        {
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )


def generate_prompt(instruction, input=None):
    return PROMPT_TEMPLATE.format(instruction=instruction)

def evals():
    problems = read_problems()
    task_ids = sorted(problems.keys())[args.start_index: args.end_index]
    prompts = [problems[task_id]['prompt'] for task_id in task_ids]
    num_samples = len(prompts)

    print("Number of samples: {}".format(num_samples))
    print(f"Loaded {args.model}.")

    generation_config = GenerationConfig(
        temperature=args.temperature,
        top_p=0.95,
    )
    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = args.output_path + '/{}.jsonl'.format(args.start_index + i)
        if args.start_index + i == 21 or args.start_index + i == 84:
            continue
        if os.path.exists(output_file) and not args.overwrite:
            print(f'Skip {output_file} as it already exists')
            continue

        completion_seqs = []

        # loops = int(args.N / args.num_seqs_per_iter)
        instruction = prompts[i]
        inputs = tokenizer(generate_prompt(
            instruction, None), return_tensors="pt")
        for _ in tqdm(range(args.N), total=args.N, leave=False, ncols=0):
            completion_seq = get_completion(generation_config, args.max_len, inputs)
            completion_seqs.append(
                {'task_id': task_ids[i],
                 'completion': completion_seq,
                 }
            )

        print("Saving results to {}".format(output_file))
        write_jsonl(output_file, completion_seqs)

def get_completion(generation_config, max_length, inputs):
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs["input_ids"].to(device),
                                 generation_config=generation_config,
                                 do_sample=True,
                                 max_new_tokens=max_length,
                                 return_dict_in_generate=True,
                                 output_scores=True)
    input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[
        1]
    generated_tokens = outputs.sequences[:, input_length:]
    return tokenizer.decode(generated_tokens[0]).replace(DEFAULT_EOS_TOKEN, "")


def gradio_inference(
        instruction,
        temperature=args.temperature,
        top_p=0.95,
        num_beams=4,
        max_new_tokens=512,
        **kwargs,
):
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
    )
    return get_completion(generation_config, max_new_tokens, instruction)

def run_interface():
    g = gr.Interface(
        fn=gradio_inference,
        inputs=[
            gr.components.Textbox(
                lines=2, label="Instruction", placeholder="Return the sum of two integers."
            ),
            gr.components.Slider(minimum=0, maximum=1,
                                 value=0.1, label="Temperature"),
            gr.components.Slider(minimum=0, maximum=1,
                                 value=0.75, label="Top p"),
            gr.components.Slider(minimum=1, maximum=4, step=1,
                                 value=4, label="Beams"),
            gr.components.Slider(minimum=1, maximum=512, step=1, value=128, label="Max tokens"),
        ],
        outputs=[
            gr.inputs.Textbox(
                lines=5,
                label="Output",
            )
        ],
        title="CoLLaMA",
        description="CoLLaMA is a Multilingual Instruction Dataset on Code and trained on large language models.",
    )
    g.queue(concurrency_count=1)
    g.launch(share=True)

if __name__ == '__main__':
    initialize()
    if args.launch_gradio:
        run_interface()
    else:
        evals()
