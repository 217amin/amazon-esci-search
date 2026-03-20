import pandas as pd
import torch, os
from pathlib import Path
from typing import Dict, Any
from sentence_transformers import (
    SentenceTransformer, 
    SentenceTransformerTrainer, 
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MatryoshkaLoss, MultipleNegativesRankingLoss
from sentence_transformers.training_args import BatchSamplers 
from datasets import Dataset 

# Workaround for Tokenizers Parallelism warning on Windows
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FixedSentenceTransformerTrainer(SentenceTransformerTrainer):
    """
    CRITICAL FIX: Overrides compute_loss to handle the 'num_items_in_batch' argument
    introduced in newer Transformers versions, which breaks Sentence-Transformers training.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return super().compute_loss(model, inputs, return_outputs)

def train_matryoshka(pair_df: pd.DataFrame, cfg: Dict[str, Any]):
    """
    Fine-tunes a BGE model using Matryoshka Representation Learning (MRL).
    Objective: Create nested embeddings where the first N dimensions are as informative as the full vector.
    """
    model_name = cfg["biencoder_model"]
    m_params = cfg["matryoshka"]
    output_dir = str(Path(cfg["paths"]["matryoshka_dir"]) / "us")
    
    print(f" Loading Backbone: {model_name}")
    print(f" Config: Batch={m_params['batch_size']} | Epochs={m_params['epochs']} | LR={m_params['lr']}")
    print(f" Max Seq Length: {m_params['max_seq_length']}")
    model = SentenceTransformer(model_name)
    model.max_seq_length = int(m_params['max_seq_length']) 
    
    # Loss Function: MRL + MNRL
    # MNRL uses in-batch negatives (efficient). MRL applies this loss at multiple vector truncations.
    inner_loss = MultipleNegativesRankingLoss(model)
    train_loss = MatryoshkaLoss(
        model=model, 
        loss=inner_loss, 
        matryoshka_dims=m_params["dims"],
        matryoshka_weights=[1.0] * len(m_params["dims"])
    )
    
    print("--> Filtering data for Training...")
    # Engineering Decision: Training on E (Exact) + S (Substitute) increases data volume.
    # While E-only is higher precision, E+S helps the model learn broader semantic associations.
    train_df = pair_df[
        (pair_df["split"] == "train") & 
        (pair_df["esci_label"].isin(["E", "S"])) 
    ].copy()
    
    # Formatting for MNRL: Needs (Anchor, Positive) pairs only.
    train_dataset = Dataset.from_dict({
        "sentence_A": ["Represent this sentence for searching relevant passages: " + q for q in train_df["query"]],
        "sentence_B": train_df["product_text_dense"].tolist(),
    })
    
    args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=int(m_params["epochs"]),
        per_device_train_batch_size=int(m_params["batch_size"]),
        learning_rate=float(m_params["lr"]),
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(), # Auto-detect mixed precision
        gradient_checkpointing=True,    
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="no", 
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none"
    )
    
    trainer = FixedSentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss
    )
    
    print("--> Starting Training...")
    trainer.train()
    trainer.save_model(output_dir)