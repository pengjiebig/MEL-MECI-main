import torch
import json
import os
import numpy as np
import requests
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

os.environ['OPENAI_API_KEY'] = "sk-key"


class ReRankingModule:
    def __init__(self, args):
        self.args = args

        self.run_name = f"KGMEL-{self.args.run_name}"
        if 'Diverse' in self.args.run_name:
            self.num = 'Y_1'
        elif 'Rich' in self.args.run_name:  # 11
            self.num = 'Y_1'
        elif 'WikiMEL' in self.args.run_name:  # 30
            self.num = 'Y_1'

        self.candidate_path = f"./logs/{self.run_name}/{self.num}/candidate-{self.args.data.num_candidates}.json"
        self.json_path = f".{self.args.data.test_file}"
        self.rerank_json_path = f"./logs/{self.run_name}/{self.num}/candidate-{self.args.data.num_candidates}_result.json"

        with open(f".{self.args.data.qid2id}", 'r', encoding='utf-8') as f:
            self.qid2id = json.loads(f.readline())
        self.qid2id = {k: str(v) for k, v in self.qid2id.items()}
        self.k_values = [1, 3, 5, 10, 16]

    def rerank(self):
        self.load_candidate_entity()
        self.evaluate_retrieval()
        self.load_data()
        self.load_mapping()

        self.preprare_rerank_data()
        return self.gpt_rerank()

    def load_candidate_entity(self):
        print('--------- Loading Candidate Entities ---------')
        with open(self.candidate_path, 'r') as f:
            self.preds = json.load(f)
            self.test_preds = self.preds['test']
        print(f"Loaded Candidate entities from {self.candidate_path}")

    def evaluate_retrieval(self):
        print('--------- Evaluating Retrieval ---------')
        retrieval_rate = {'test': 0}
        for split in ['test']:
            cnt = sum(a in c for a, c in zip(self.preds[split]['answer'], self.preds[split]['candidate']))
            retrieval_rate[split] = 100 * cnt / len(self.preds[split]['mention_key'])

    def load_data(self):
        print('--------- Loading Data ---------')
        with open(self.json_path, 'r') as f:
            self.test_data = json.load(f)

    def load_mapping(self):
        print('--------- Loading Mapping ---------')
        # for Entity
        kb_entity_path = f".{self.args.data.entity}"
        with open(kb_entity_path, 'r') as f:
            kb_entities = json.load(f)

        self.qid2desc = {}
        self.qid2label = {}
        for entity in kb_entities:
            qid = str(entity['id'])
            self.qid2desc[qid] = entity['desc']
            self.qid2label[qid] = entity['entity_name']

        self.mention_key2cand, self.mention_key2rank = {}, {}
        for mention_key, candidate, rank in zip(self.test_preds['mention_key'], self.test_preds['candidate'],
                                                self.test_preds['rank']):
            self.mention_key2cand[mention_key] = candidate
            self.mention_key2rank[mention_key] = rank

    def process_candidate(self, candidate):
        cand_data = []
        for i, c in enumerate(candidate):
            cand_data.append({'qid': c, 'label': self.qid2label[c], 'desc': self.qid2desc[c]})
        return cand_data

    def preprare_rerank_data(self):
        print('--------- Preparing Rerank Data ---------')
        self.rerank_test_data = []
        for test_item in self.test_data:
            for a, m, e in zip([test_item['answer']], [test_item['mentions']], [test_item['entities']]):
                item = ({'id': test_item['id'],
                         'sentence': test_item['sentence'],
                         'mention': m, 'answer': self.qid2id[a], 'label': e,
                         'mention-desc': test_item['desc'],
                         'retrieve-hit': None,
                         'retrieve-rank': self.mention_key2rank[f"{test_item['id']}-{self.qid2id[a]}"],
                         'candidate': []})

                candidate = self.mention_key2cand[f"{item['id']}-{self.qid2id[a]}"]
                if self.qid2id[a] not in candidate:
                    item['hit'] = 0
                    self.rerank_test_data.append(item)
                    continue
                item['candidate'] = self.process_candidate(candidate)

                self.rerank_test_data.append(item)

        with open(self.rerank_json_path, 'w') as f:
            json.dump(self.rerank_test_data, f, indent=2)

    def format_prompt_gpt(self, item):
        candidate = [f"{t['label']} (Q{t['qid']}):{t['desc']}" for t in item['candidate']][::-1]
        candidate = '\n'.join(candidate)
        return f"""Given the context below, please identify the most corresponding entity from the list of candidates.

Context: {item['sentence']}

Candidate Entities:
{candidate}

Target Entity: "{item['mention']}": {item['mention-desc']}

Based on the context and entity description, identify the most relevant entity that best matches the given sentence context. Please provide a clear answer with the entity QID."""

    def verify_response(self, response, label, answer):
        # for LLaMA response often contains multiple QIDs we use the first QID
        qid = re.search(r'Q(\d+)', response)
        if qid:
            # Extract the numeric part of the QID to compare with answer
            if qid.group(1) == answer:
                return 1
        # Check if the label appears in the response (case insensitive)
        if label.lower() in response.lower():
            return 1
        return 0

    def evaluate(self, ranks):
        # Print Hit@k metrics for each k value in self.k_values
        for k in self.k_values:
            hits = sum([1 for r in ranks if r <= k])
            hit_rate = 100 * hits / len(ranks)
            print(f"H@{k}: {hit_rate:.2f}", end=" ")

        # Calculate and print Mean Reciprocal Rank (MRR)
        mrr = 100 * np.mean([1 / r for r in ranks])
        print(f"MRR: {mrr:.2f}")

        # Prepare results dictionary
        result = {"MRR": mrr}

        # Add Hit@k metrics to results dictionary
        result.update({
            f"H@{k}": 100 * sum([1 for r in ranks if r <= k]) / len(ranks)
            for k in self.k_values
        })

        return result

    def gpt_rerank(self):
        print('--------- Reranking with GPT ---------')

        # Initialize API settings
        api_key = os.environ.get("OPENAI_API_KEY")
        api_url = "https://api.hello-ai.cn/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

        # Initialize rank tracking lists
        ranks, rerank_ranks = [], []

        # Process each test item
        for i, item in enumerate(self.rerank_test_data):
            # --- Initialize reranking fields ---
            self.rerank_test_data[i]['rerank-response'] = ''
            self.rerank_test_data[i]['rerank-hit'] = 0
            self.rerank_test_data[i]['rerank-rank'] = item['retrieve-rank']

            # --- Skip items where answer is not in candidates ---
            if item['retrieve-hit'] == 0:
                ranks.append(self.rerank_test_data[i]['retrieve-rank'])
                rerank_ranks.append(self.rerank_test_data[i]['rerank-rank'])
                print(f"Answer NOT in Candidate")
                continue

            # --- Get LLM response ---
            try:
                # Generate completion using OpenAI API
                payload = {
                    "model": self.args.data.llm,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that helps users find the most relevant entity from a list of candidates."
                        },
                        {
                            "role": "user",
                            "content": self.format_prompt_gpt(item)
                        }
                    ],
                    "max_tokens": 4000
                }

                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                response_data = response.json()
                response_content = response_data["choices"][0]["message"]["content"]

                # Store response
                self.rerank_test_data[i]['rerank-response'] = response_content

                # Print debug information
                print('=' * 50)
                print(f"[Answer] {item['answer']}")
                print(f"[Label] {item['label']}")
                print(f"[Response] {response_content}")

            except Exception as e:
                print(f"Error: {e}")

            # --- Evaluate response ---`
            # Check if response matches expected answer or label
            self.rerank_test_data[i]['rerank-hit'] = self.verify_response(
                self.rerank_test_data[i]['rerank-response'],
                item['label'],
                item['answer']
            )

            # Set rank position based on hit status
            self.rerank_test_data[i]['rerank-rank'] = 1 if item['rerank-hit'] == 1 else item['retrieve-rank'] + 1

            # Add to rank tracking lists
            ranks.append(item['retrieve-rank'])
            rerank_ranks.append(self.rerank_test_data[i]['rerank-rank'])

            # --- Log and save metrics ---
            # Calculate and log current metrics
            print(f"Retrieval Result:", end=" ")
            retrieve_eval = self.evaluate(ranks)
            print(f"Reranking Result:", end=" ")
            rerank_eval = self.evaluate(rerank_ranks)

            # Verification check
            assert len(ranks) == len(rerank_ranks)

            # Save intermediate results at regular intervals
            if i % 50 == 0:
                with open(self.rerank_json_path, 'w') as f:
                    json.dump(self.rerank_test_data, f)
                print(f"Saved reranking results to {self.rerank_json_path}")

        # --- Save final results ---
        with open(self.rerank_json_path, 'w') as f:
            json.dump(self.rerank_test_data, f)
        print(f"Saved reranking results to {self.rerank_json_path}")

        # Calculate final evaluation metrics
        rerank_eval = self.evaluate(rerank_ranks)
        retrieve_eval = self.evaluate(ranks)

        return rerank_eval, retrieve_eval

    def format_prompt_llama(self, item):
        candidate = [f"{t['label']} ({t['qid']}):{t['desc']}" for t in item['candidate']][::-1]
        candidate = '\n'.join(candidate)
        return f"""<s>[INST] <<SYS>>
You are a helpful assistant that helps users find the most relevant entity from a list of candidates.
<</SYS>>

Given the context below, please identify the most corresponding entity from the list of candidates.

Context: {item['sentence']}

Candidate Entities:
{candidate}

Target Entity: "{item['mention']}": {item['mention-desc']}

Based on the context and entity description, identify the most relevant entity that best matches the given sentence context. Please provide ONLY ONE entity with its QID which is the most relevant to the given context and target entity.[/INST]"""

    def llama_rerank(self):
        print('--------- Reranking with LLaMA ---------')

        login(token=os.environ.get("HUGGINGFACE_TOKEN"))

        # Initialize model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.data.llm)
        model = AutoModelForCausalLM.from_pretrained(self.args.data.llm,
                                                     torch_dtype=torch.float16,
                                                     # cache_dir=self.args.huggingface_cache_dir,
                                                     use_auth_token=True,
                                                     device_map="auto").to(self.args.device)

        # Initialize rank tracking lists
        ranks, rerank_ranks = [], []

        # Process each test item
        for i, item in enumerate(self.rerank_test_data):
            # --- Initialize reranking fields ---
            self.rerank_test_data[i]['rerank-response'] = ''
            self.rerank_test_data[i]['rerank-hit'] = 0
            self.rerank_test_data[i]['rerank-rank'] = item['retrieve-rank']

            # --- Skip items where answer is not in candidates ---
            if item['retrieve-hit'] == 0:
                ranks.append(self.rerank_test_data[i]['retrieve-rank'])
                rerank_ranks.append(self.rerank_test_data[i]['rerank-rank'])
                print(f"Answer NOT in Candidate")
                continue

            # --- Get LLM response ---
            try:
                # Generate completion using OpenAI API
                inputs = tokenizer(self.format_prompt_llama(item), return_tensors="pt").to("cuda")
                output = model.generate(**inputs, max_new_tokens=512)
                response = tokenizer.decode(output[0], skip_special_tokens=True).split('[/INST]')[-1].strip()

                # Store response
                self.rerank_test_data[i]['rerank-response'] = response

                # Print debug information
                print('=' * 50)
                print(f"[Answer] {item['answer']}")
                print(f"[Label] {item['label']}")
                print(f"[Response] {response}")

            except Exception as e:
                print(f"Error: {e}")

            # --- Evaluate response ---
            # Check if response matches expected answer or label
            self.rerank_test_data[i]['rerank-hit'] = self.verify_response(
                self.rerank_test_data[i]['rerank-response'],
                item['label'],
                item['answer']
            )

            # Set rank position based on hit status
            self.rerank_test_data[i]['rerank-rank'] = 1 if item['rerank-hit'] == 1 else item['retrieve-rank'] + 1

            # Add to rank tracking lists
            ranks.append(item['retrieve-rank'])
            rerank_ranks.append(self.rerank_test_data[i]['rerank-rank'])

            # --- Log and save metrics ---
            # Calculate and log current metrics
            print(f"Retrieval Result:", end=" ")
            retrieve_eval = self.evaluate(ranks)
            print(f"Reranking Result:", end=" ")
            rerank_eval = self.evaluate(rerank_ranks)

            # Verification check
            assert len(ranks) == len(rerank_ranks)

            # Save intermediate results at regular intervals
            if i % 50 == 0:
                with open(self.rerank_json_path, 'w') as f:
                    json.dump(self.rerank_test_data, f)
                print(f"Saved reranking results to {self.rerank_json_path}")

        # --- Save final results ---
        with open(self.rerank_json_path, 'w') as f:
            json.dump(self.rerank_test_data, f)
        print(f"Saved reranking results to {self.rerank_json_path}")

        # Calculate final evaluation metrics
        rerank_eval = self.evaluate(rerank_ranks)
        retrieve_eval = self.evaluate(ranks)

        return rerank_eval, retrieve_eval


