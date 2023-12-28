from datasets import load_dataset

import logging
logging.basicConfig(level='ERROR')

import argparse
import numpy as np
import pandas as pd
import sys
import os
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

class ExperimentArgs:
    def __init__(self,
                 N=None,
                 k=1,
                 n_samples=20,
                 temperature=0.2,
                 top_p=0.95,
                 checkpoint="smallcloudai/Refact-1_6B-fim"):
        """
        N : int -- The number of samples to process. If None, will use the length of the dataset.
        k : int -- The number of code candidates to consider in the evaluation.
        n_samples : int -- The number of samples to generate for each prompt.
        temperature : float -- The temperature parameter for generation, controlling randomness.
        top_p : float -- The top-p parameter for generation, controlling diversity.
        checkpoint : str -- The model checkpoint path.
        model_name : str -- The name of the model, extracted from the checkpoint path.
        """
        self.N = N
        self.k = k
        self.n_samples = n_samples
        self.temperature = temperature
        self.top_p = top_p
        self.checkpoint = checkpoint
        self.model_name = os.path.split(checkpoint)[1]

class Experiment:
    def __init__(self, filelog, filetable, args : ExperimentArgs):
        self.args = args
        self.filelog = filelog
        self.filetable = filetable
        self.pass_at_k = evaluate.load("code_eval")
        self.score_name = f"pass@{self.args.k}" 

        df = pd.DataFrame(columns=['text', self.score_name])
        df.to_csv(self.filetable, mode='a', index=False)

    def setup(self):
        with open(self.filelog, 'a') as f:
            f.write(f"using device: {device}\n")
            f.write("Loading Model...\n\n")

        self.dataset = load_dataset("bigcode/humanevalpack", "python")["test"]
        if self.args.N is None:
            self.args.N = len(self.dataset)
        self.prompt_template = "<empty_output>SYSTEM {system}\n" \
                               "<empty_output>USER {query}\n" \
                               "<empty_output>ASSISTANT"

        self.tokenizer = AutoTokenizer.from_pretrained(self.args.checkpoint)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.checkpoint,
            trust_remote_code=True).to(device)
        self.model.eval()
        with open(self.filelog, 'a') as f:
            f.write("Model loading is done!\n\n")

        self.scores = {
            self.score_name: []
        }

    def generate_sequences(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        outputs = self.model.generate(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            max_length=2*len(prompt),
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            num_return_sequences=self.args.n_samples)
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    def get_code(self, sequence):
        parts = sequence.split("ASSISTANT")
        if len(parts) > 1:
            code = parts[1].strip()
        else:
            code = ""

        return code

    def run(self):
        with open(self.filelog, 'a') as f:
            f.write("Start experiment...\n\n")

        for i in tqdm(range(self.args.N)):
            item = self.dataset[i]
            prompt = self.prompt_template.format(
                system="You are a programming assistant",
                query="Fix bugs in " + item['entry_point'] + "\n" + item['declaration'] + item['buggy_solution'])

            generated_sequences = self.generate_sequences(prompt)
            generated_code = [self.get_code(seq) for seq in generated_sequences]

            score, _ = self.pass_at_k.compute(references=[item['test']], predictions=[generated_code], k=[self.args.k])
            self.scores[self.score_name].append(score[self.score_name])

            data = {
                'text': ["\n\n".join(generated_sequences)],
                self.score_name: [score[self.score_name]],
            }
            df = pd.DataFrame(data)
            df.to_csv(self.filetable, mode='a', header=False, index=False)

        self.scores[self.score_name] = np.asarray(self.scores[self.score_name])
        with open(self.filelog, 'a') as f:
            f.write(f"{self.score_name} = {self.scores[self.score_name].mean()}\n")
            f.write("Experiment done!\n\n")


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=None, help="The number of samples to process. If None, will use the length of the dataset.")
    parser.add_argument('--k', type=int, default=1, help="The number of code candidates to consider in the evaluation.")
    parser.add_argument('--n_samples', default=20, help="The number of samples to generate for each prompt.")
    parser.add_argument('--temperature', type=float, default=0.2, help="The temperature parameter for generation, controlling randomness.")
    parser.add_argument('--top_p', type=float, default=0.95, help="The top-p parameter for generation, controlling diversity.")
    parser.add_argument('--checkpoint', type=str, default="smallcloudai/Refact-1_6B-fim", help="The name of the model, extracted from the checkpoint path.")
    parser.add_argument('--filelog', type=str, default="exp-log-v1.txt", help="The file name for logging.")
    parser.add_argument('--filetable', type=str, default="exp-table-v1.txt", help="The file name for logging.")

    return parser.parse_args(argv)

def main():
    experiment_args = ExperimentArgs(N=args.N, 
                                     k=args.k, 
                                     n_samples=args.n_samples, 
                                     temperature=args.temperature, 
                                     top_p=args.top_p, 
                                     checkpoint=args.checkpoint)
    experiment = Experiment(filelog=args.filelog, filetable=args.filetable, args=experiment_args)
    experiment.setup()
    experiment.run()


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main()