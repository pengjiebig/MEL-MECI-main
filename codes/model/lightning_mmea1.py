import numpy as np
import torch
import pytorch_lightning as pl
from tqdm import tqdm
import json
import os
from pathlib import Path
import tempfile
import shutil
from codes.model.modeling_mmea import CLIPBackboneWrapper, RetrievalModel

PROJECT_ROOT = Path(__file__).resolve().parent
EMBED_CACHE_ROOT = PROJECT_ROOT / "embed_cache"
EMBED_CACHE_ROOT.mkdir(exist_ok=True)


class LightningForMMoE(pl.LightningModule):
    def __init__(self, args):
        super(LightningForMMoE, self).__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.encoder = CLIPBackboneWrapper(args)
        self.matcher = RetrievalModel(args)

        self.loss_fct = torch.nn.CrossEntropyLoss()

        self.weight_update_counter = 0
        self.weight_update_frequency = 100

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        ent_batch = {}
        mention_batch = {}
        for k, v in batch.items():
            if k.startswith('ent_'):
                ent_batch[k.replace('ent_', '')] = v
            else:
                mention_batch[k] = v

        ent_batch.pop('empty_img_flag', None)

        mention_text_embeds, mention_image_embeds, mention_text_seq_tokens, mention_image_patch_tokens = \
            self.encoder(**mention_batch)
        entity_text_embeds, entity_image_embeds, entity_text_seq_tokens, entity_image_patch_tokens = \
            self.encoder(**ent_batch)

        total_joint, (expert_logit, logit_joint_fused), router_loss = \
            self.matcher(entity_text_embeds,
                         entity_text_seq_tokens,
                         mention_text_embeds,
                         mention_text_seq_tokens,
                         entity_image_embeds,
                         entity_image_patch_tokens,
                         mention_image_embeds,
                         mention_image_patch_tokens)

        labels = torch.arange(len(mention_text_embeds)).long().to(mention_text_embeds.device)

        overall_loss = self.loss_fct(total_joint, labels)
        expert_logit = self.loss_fct(expert_logit, labels)
        logit_joint_fused = self.loss_fct(logit_joint_fused, labels)
        total_loss = overall_loss + expert_logit + logit_joint_fused + self.args.model.router_weight * router_loss
        print(self.args.model.router_weight * router_loss)
        self.log('Train/loss', total_loss.detach().cpu().item(), on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        answer = batch.pop('answer').cpu()
        mention_keys = batch.pop('mention_id').cpu()
        batch_size = len(answer)

        with torch.no_grad():
            mention_data = self.encoder(**batch)

        chunk_scores = []
        chunk_size = self.args.data.eval_chunk_size

        total_entities = self.total_entities
        self.output_size = len(mention_data)

        for start_idx in range(0, total_entities, chunk_size):
            end_idx = min(start_idx + chunk_size, total_entities)

            chunk_data = self._get_chunk_from_list_cached(start_idx, end_idx)
            if chunk_data is None:
                continue

            device = mention_data[0].device
            chunk_score, _, _ = self.matcher(
                chunk_data[0].to(device), chunk_data[2].to(device),
                mention_data[0], mention_data[2],
                chunk_data[1].to(device), chunk_data[3].to(device),
                mention_data[1], mention_data[3]
            )
            chunk_scores.append(chunk_score.cpu())

            del chunk_score
            del chunk_data

        scores = torch.concat(chunk_scores, dim=-1)
        rank = torch.argsort(torch.argsort(scores, dim=-1, descending=True), dim=-1, descending=False) + 1
        tgt_rank = rank[torch.arange(batch_size), answer].detach().cpu()
        torch.cuda.empty_cache()
        return dict(rank=tgt_rank, all_rank=rank.detach().cpu().numpy())

    def on_validation_start(self):
        entity_dataloader = self.trainer.datamodule.entity_dataloader()

        sample_batch = next(iter(entity_dataloader))
        sample_batch = pl.utilities.move_data_to_device(sample_batch, self.device)
        with torch.no_grad():
            sample_output = self.encoder(**sample_batch)

        # total_entities = sum(len(batch[list(batch.keys())[0]]) for batch in entity_dataloader)
        total_entities = len(self.trainer.datamodule.kb_entity)

        embed_shapes = [emb.shape[1:] for emb in sample_output]

        # self.embed_cache_dir = Path(tempfile.mkdtemp(prefix="entity_embeds_memmap_"))
        self.embed_cache_dir = Path(tempfile.mkdtemp(
            prefix="entity_embeds_memmap_",
            dir=EMBED_CACHE_ROOT
        ))
        self.embed_files = []
        self.embed_arrays = []

        print(f"Total entities: {total_entities}")
        print(f"Embed shapes: {embed_shapes}")

        for i, shape in enumerate(embed_shapes):
            mmap_file = self.embed_cache_dir / f"embed_{i}.dat"
            mmap_array = np.memmap(mmap_file, dtype='float32', mode='w+',
                                   shape=(total_entities, *shape))
            self.embed_files.append(mmap_file)
            self.embed_arrays.append(mmap_array)

        print("Encoding entity embeddings to memory-mapped files...")
        current_idx = 0
        with torch.no_grad():
            for batch in tqdm(entity_dataloader, desc='UpdateEmbed', total=len(entity_dataloader)):

                batch = pl.utilities.move_data_to_device(batch, self.device)
                entity_data = self.encoder(**batch)
                batch_size = len(entity_data[0])

                for i, emb in enumerate(entity_data):
                    self.embed_arrays[i][current_idx:current_idx + batch_size] = emb.cpu().numpy()

                current_idx += batch_size
                del entity_data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.total_entities = total_entities
        print(f"Encoded {current_idx} entities to memory-mapped files")

    def _get_chunk_from_list_cached(self, start_idx, end_idx):
        if start_idx >= self.total_entities:
            return None

        end_idx = min(end_idx, self.total_entities)
        chunk_data = []
        for i, embed_array in enumerate(self.embed_arrays):
            chunk = embed_array[start_idx:end_idx]
            chunk_data.append(torch.from_numpy(chunk))

        return tuple(chunk_data)

    def validation_epoch_end(self, outputs):
        self._cleanup_embed_cache()

        if hasattr(self, 'batch_sizes'):
            del self.batch_sizes
        if hasattr(self, 'cumulative_sizes'):
            del self.cumulative_sizes

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ranks = np.concatenate([_['rank'] for _ in outputs])
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits1 = (ranks <= 1).mean()

        self.log("Val/hits20", hits20)
        self.log("Val/hits10", hits10)
        self.log("Val/hits5", hits5)
        self.log("Val/hits3", hits3)
        self.log("Val/hits1", hits1)
        self.log("Val/mr", ranks.mean())
        self.log("Val/mrr", (1. / ranks).mean())

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        answer = batch.pop('answer').cpu()
        mention_keys = batch.pop('mention_id').cpu()
        batch_size = len(answer)

        with torch.no_grad():
            mention_data = self.encoder(**batch)

        chunk_scores = []
        chunk_size = self.args.data.eval_chunk_size
        total_entities = self.total_entities

        for start_idx in range(0, total_entities, chunk_size):
            end_idx = min(start_idx + chunk_size, total_entities)

            chunk_data = self._get_chunk_from_list_cached(start_idx, end_idx)
            if chunk_data is None:
                continue

            device = mention_data[0].device
            chunk_score, _, _ = self.matcher(
                chunk_data[0].to(device), chunk_data[2].to(device),
                mention_data[0], mention_data[2],
                chunk_data[1].to(device), chunk_data[3].to(device),
                mention_data[1], mention_data[3]
            )
            chunk_scores.append(chunk_score.cpu())

            del chunk_score
            del chunk_data

        scores = torch.concat(chunk_scores, dim=-1)
        rank = torch.argsort(torch.argsort(scores, dim=-1, descending=True), dim=-1, descending=False) + 1
        tgt_rank = rank[torch.arange(batch_size), answer].detach().cpu()

        top_k = min(16, scores.shape[1])
        top_k_scores, top_k_indices = torch.topk(scores, k=top_k, dim=1)

        torch.cuda.empty_cache()
        return dict(
            rank=tgt_rank,
            all_rank=rank.detach().cpu().numpy(),
            scores=scores.detach().cpu().numpy(),
            mention_keys=mention_keys,
            answer=answer,
            top_k_indices=top_k_indices.detach().cpu().numpy()
        )

    def on_test_start(self):
        self.candidate_preds = {
            'answer': [],
            'mention_key': [],
            'candidate': [],
            'rank': []
        }
        if not hasattr(self, 'embed_arrays') or not self.embed_arrays:
            self.on_validation_start()

    def test_epoch_end(self, outputs):
        self._cleanup_embed_cache()

        if hasattr(self, 'batch_sizes'):
            del self.batch_sizes
        if hasattr(self, 'cumulative_sizes'):
            del self.cumulative_sizes

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        ranks = np.concatenate([_['rank'] for _ in outputs])
        hits20 = (ranks <= 20).mean()
        hits10 = (ranks <= 10).mean()
        hits5 = (ranks <= 5).mean()
        hits3 = (ranks <= 3).mean()
        hits1 = (ranks <= 1).mean()

        self.log("Test/hits20", hits20)
        self.log("Test/hits10", hits10)
        self.log("Test/hits5", hits5)
        self.log("Test/hits3", hits3)
        self.log("Test/hits1", hits1)
        self.log("Test/mr", ranks.mean())
        self.log("Test/mrr", (1. / ranks).mean())

        # Fill candidate_preds
        total_samples = 0
        for output in outputs:
            mention_keys = output['mention_keys']
            top_k_indices = output['top_k_indices']
            answers = output['answer'].cpu().numpy() if torch.is_tensor(output['answer']) else np.array(
                output['answer'])
            all_ranks = output['all_rank']

            for i in range(len(answers)):
                ans = int(answers[i])
                mk = str(mention_keys[i].item())
                cands = top_k_indices[i]
                r = int(all_ranks[i][ans])

                self.candidate_preds['answer'].append(str(ans))
                self.candidate_preds['mention_key'].append(f"{mk}-{ans}")
                self.candidate_preds['candidate'].append([str(idx) for idx in cands])
                self.candidate_preds['rank'].append(r)
                total_samples += 1

        print(f"Collected candidate predictions for {total_samples} samples.")
        print("Final candidate_preds lengths:", {k: len(v) for k, v in self.candidate_preds.items()})

        self._save_candidate_file()

    def _save_candidate_file(self):
        run_name = f"KGMEL-{self.args.run_name}"
        candidate_dir = f"./logs/{run_name}"
        os.makedirs(candidate_dir, exist_ok=True)
        candidate_path = f"{candidate_dir}/candidate-{self.args.data.num_candidates}.json"

        final_candidate_preds = {
            'test': self.candidate_preds
        }
        try:
            with open(candidate_path, 'w', encoding='utf-8') as f:
                json.dump(final_candidate_preds, f, indent=4, ensure_ascii=False)
            print(f"✅ Successfully saved candidate predictions to {candidate_path}")
        except Exception as e:
            print(f"❌ Failed to save candidate file: {e}")

    def _cleanup_embed_cache(self):
        if hasattr(self, 'embed_cache_dir') and self.embed_cache_dir.exists():
            shutil.rmtree(self.embed_cache_dir)
        if hasattr(self, 'embed_arrays'):
            del self.embed_arrays
        if hasattr(self, 'embed_files'):
            del self.embed_files
        if hasattr(self, 'total_entities'):
            del self.total_entities

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_params = [
            {'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0001},
            {'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_params, lr=self.args.lr, betas=(0.9, 0.999), eps=1e-4)
        return [optimizer]


