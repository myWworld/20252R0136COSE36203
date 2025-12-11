
import os
import glob
import pandas as pd

from sentiment_utils import (
    load_dataset, split_train_valid_test, ClassicCfg, ClassicSentiment,
    apply_neutral_policy, small_grid_search, merge_gold_and_weak,
    batch_predict, chi2_top_ngrams_from_df, ensure_cols
)

class CFG:
    DATA_LABELED_BASE = 'comments_labeled.csv'
    DATA_LABELED_MERGED = 'comments_labeled_merged.csv'
    LABELS = ['neg','neu','pos']
    TEST_SIZE = 0.1
    VALID_SIZE = 0.1
    RANDOM_STATE = 42
    ART_DIR = 'artifacts/classic'
    EXPORT_DIR = 'exports_ngrams'
    NEW_CSV = 'comments_new.csv'  # optional

def ensure_dirs():
    os.makedirs(CFG.ART_DIR, exist_ok=True)
    os.makedirs(CFG.EXPORT_DIR, exist_ok=True)

def merge_all_manual_golds(base_path: str, out_path: str, weak_frac: float = 0.15):
    # Load base labeled
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Missing base labeled file: {base_path}")
    df_weak = load_dataset(base_path, labels=CFG.LABELS)

    # Load all manual_labeled_round*.csv
    gold_paths = sorted(glob.glob('manual_labeled_round*.csv'))
    if not gold_paths:
        print("[merge] No manual_labeled_round*.csv found. Using base only.")
        df_weak.to_csv(out_path, index=False, encoding='utf-8-sig')
        return out_path

    gold_list = []
    import re
    for p in gold_paths:
        d = pd.read_csv(p, encoding='utf-8')
        ensure_cols(d, ['text','label'])
        d['text'] = d['text'].astype(str).str.strip()
        d['label'] = d['label'].astype(str).str.lower().str.strip()
        m = re.search(r'round(\d+)', p)
        d['round'] = int(m.group(1)) if m else 0
        gold_list.append(d)
    df_gold_all = pd.concat(gold_list, ignore_index=True)

    key = 'comment_id' if ('comment_id' in df_weak.columns and 'comment_id' in df_gold_all.columns) else 'text'
    # Keep latest round for duplicate keys
    df_gold_all = df_gold_all.sort_values(by=['round']).drop_duplicates(subset=[key], keep='last')
    # Remove overlapping weak rows
    df_weak = df_weak[~df_weak[key].isin(set(df_gold_all[key]))]
    # Downsample weak
    if len(df_weak) and weak_frac < 1.0:
        df_weak = df_weak.sample(frac=weak_frac, random_state=CFG.RANDOM_STATE)
    df_merged = pd.concat([df_gold_all, df_weak], ignore_index=True)
    df_merged.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[merge] Saved merged -> {out_path} (size={len(df_merged)})")
    return out_path

def baseline_once(data_path: str):
    df = load_dataset(data_path, labels=CFG.LABELS)
    train_df, valid_df, test_df = split_train_valid_test(df, CFG.TEST_SIZE, CFG.VALID_SIZE, CFG.RANDOM_STATE)

    train_texts = train_df['text'].tolist()
    valid_texts = valid_df['text'].tolist()
    test_texts  = test_df['text'].tolist()
    y_train, y_valid, y_test = train_df['label'].values, valid_df['label'].values, test_df['label'].values

    best_cfg, best_tau, best_score = small_grid_search(train_texts, y_train, valid_texts, y_valid)
    print(f'[Baseline] Best valid macro-F1={best_score:.4f} | cfg={best_cfg} | tau={best_tau}')

    final_model = ClassicSentiment(best_cfg).fit(train_texts + valid_texts, list(y_train) + list(y_valid))
    proba_test, labels_test = final_model.predict_proba(test_texts)
    pred_test = apply_neutral_policy(proba_test, labels_test, tau=best_tau, gap=0.05)

    from sklearn.metrics import classification_report, confusion_matrix
    print("=== Test Report (final) ===")
    print(classification_report(y_test, pred_test, digits=4))
    print("Confusion Matrix:\\n", confusion_matrix(y_test, pred_test, labels=CFG.LABELS))

    final_model.save(CFG.ART_DIR)
    print(f"[baseline] Artifacts saved -> {CFG.ART_DIR}")
    return best_tau

def optional_outputs(best_tau):
    # Batch inference if NEW_CSV exists
    if os.path.exists(CFG.NEW_CSV):
        out = batch_predict(CFG.NEW_CSV, CFG.ART_DIR, tau=best_tau, gap=0.05)
        print("[inference] Saved ->", out)
        df_pred = pd.read_csv(out, encoding='utf-8')
        chi2_top_ngrams_from_df(df_pred, label_col='pred', export_dir=CFG.EXPORT_DIR)
        print("[chi2] Exported top n-grams ->", CFG.EXPORT_DIR)

def main():
    ensure_dirs()
    merged_path = merge_all_manual_golds(CFG.DATA_LABELED_BASE, CFG.DATA_LABELED_MERGED, weak_frac=0.01)
    tau = baseline_once(merged_path)
    optional_outputs(tau)
    print("[Done] Submission pipeline finished.")

if __name__ == "__main__":
    main()
