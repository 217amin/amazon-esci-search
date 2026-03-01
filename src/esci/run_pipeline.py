import argparse, yaml
import pandas as pd
from src.esci.data import sample_dataset, add_product_text, add_grades_and_pair_view
from src.esci.matryoshka_train import train_matryoshka
from src.esci.system_a import encode_systemA
from src.esci.system_b import build_candidates, rerank_candidates

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "inference"], required=True)
    args = parser.parse_args()

    with open("configs/esci.yaml") as f:
        cfg = yaml.safe_load(f)

    print("Loading Data...")
    df = pd.read_parquet(cfg["paths"]["raw_examples"])
    df = sample_dataset(df, cfg) 
    df = add_product_text(df)
    df = add_grades_and_pair_view(df)

    if args.mode == "train":
        # Corrected function call
        train_matryoshka(df, cfg)
    elif args.mode == "inference":
        encode_systemA(df, cfg)
        cands, ret_qps = build_candidates(cfg)
        final, rerank_qps = rerank_candidates(cands, cfg)
        print("Done. Top 5:\n", final.head())

if __name__ == "__main__":
    main()