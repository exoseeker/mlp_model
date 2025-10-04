import argparse, os, json, warnings, sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Compact, high-signal Kepler features
KEPLER_FEATURES = [
    "koi_period", "koi_duration", "koi_depth", "koi_model_snr", "koi_impact",
    "koi_prad", "koi_teq", "koi_insol",
    "koi_steff", "koi_slogg", "koi_srad",
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
]
LABEL_COL = "koi_disposition"
MULTI_CLASSES = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]

def load_csv_smart(path: str) -> pd.DataFrame:
    for sep in [",", "\t"]:
        try:
            df = pd.read_csv(path, sep=sep, comment="#", engine="python")
            if df.shape[1] > 1:
                return df
        except Exception:
            pass
    raw = [ln for ln in open(path, "r", encoding="utf-8", errors="ignore").read().splitlines()
           if not ln.startswith("#") and ln.strip()]
    header = raw[0]
    delim = "," if header.count(",") >= header.count("\t") else "\t"
    data = [r.split(delim) for r in raw]
    return pd.DataFrame(data[1:], columns=data[0])

def prepare_kepler(df: pd.DataFrame, binary: bool):
    feats = [c for c in KEPLER_FEATURES if c in df.columns]
    need = [LABEL_COL] + feats
    df = df.dropna(subset=[LABEL_COL])
    df = df[[c for c in need if c in df.columns]].copy()

    # numeric coercion + median impute
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().all():
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(df[c].median())

    if binary:
        # 1 = PLANET (CONFIRMED or CANDIDATE), 0 = NOT (FALSE POSITIVE)
        df = df[df[LABEL_COL].isin(MULTI_CLASSES)].copy()
        y = np.where(df[LABEL_COL].isin(["CONFIRMED", "CANDIDATE"]), 1, 0).astype(int)
        class_names = ["NOT", "PLANET"]
    else:
        df = df[df[LABEL_COL].isin(MULTI_CLASSES)].copy()
        class_to_idx = {cls: i for i, cls in enumerate(MULTI_CLASSES)}
        y = df[LABEL_COL].map(class_to_idx).astype(int).values
        class_names = MULTI_CLASSES

    X = df[feats].astype(np.float32).values
    return X, y, feats, class_names

# ---------- MLP path ----------
class TabularDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x): return self.net(x)

def train_mlp(X_tr, y_tr, X_te, y_te, class_names, epochs=60, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # class weights
    classes = np.unique(y_tr)
    weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr).astype(np.float32)
    class_weights = torch.from_numpy(weights_np).to(device=device, dtype=torch.float32)

    train_loader = DataLoader(TabularDS(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(TabularDS(X_te, y_te), batch_size=max(256, batch_size), shuffle=False)

    model = MLP(in_dim=X_tr.shape[1], out_dim=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)

    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(DataLoader(TabularDS(X_tr, y_tr), batch_size=batch_size, shuffle=True),
                    desc=f"Epoch {epoch}/{epochs}", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * yb.size(0)
            pbar.set_postfix(avg_loss=f"{running/len(y_tr):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.1e}")

        # quick val
        model.eval()
        preds, probs = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                logits = model(xb.to(device))
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(p)
                preds.append(p.argmax(1))
        y_pred = np.concatenate(preds)
        y_prob = np.vstack(probs)
        acc = accuracy_score(y_te, y_pred)
        scheduler.step(acc)
        tqdm.write(f"Epoch {epoch:02d} | val_acc={acc:.4f}")

        if acc > best_acc + 1e-4:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v for k, v in best_state.items()})

    # final eval
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1).cpu().numpy()
            probs.append(p)
            preds.append(p.argmax(1))
    y_pred = np.concatenate(preds)
    y_prob = np.vstack(probs)

    print("\n[MLP] Test accuracy:", accuracy_score(y_te, y_pred))
    print("\n[MLP] Classification report:")
    print(classification_report(y_te, y_pred, target_names=class_names))
    print("\n[MLP] Confusion matrix:\n", confusion_matrix(y_te, y_pred))
    return ("mlp", model, y_pred, y_prob, best_acc)

# ---------- Tree path ----------
def train_tree(X_tr, y_tr, X_te, y_te, class_names):
    clf = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_depth=8,
        max_iter=800,
        l2_regularization=0.0,
        min_samples_leaf=20,
        random_state=SEED
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)
    acc = accuracy_score(y_te, y_pred)
    print("\n[Tree] Test accuracy:", acc)
    print("\n[Tree] Classification report:")
    print(classification_report(y_te, y_pred, target_names=class_names))
    print("\n[Tree] Confusion matrix:\n", confusion_matrix(y_te, y_pred))
    return ("tree", clf, y_pred, y_prob, acc)

def main(csv_path: str, binary: bool, model_choice: str, epochs: int, batch_size: int):
    print("numpy:", np.__version__)
    print(f"Loading {csv_path}")
    df = load_csv_smart(csv_path)

    X, y, features, class_names = prepare_kepler(df, binary=binary)

    # 79 / 21 split (stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.21, random_state=SEED, stratify=y
    )

    # Scale: trees don't need it, but it doesn't hurt; MLP benefits.
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    if model_choice == "tree":
        kind, model, y_pred, y_prob, acc = train_tree(X_tr, y_tr, X_te, y_te, class_names)
    else:
        kind, model, y_pred, y_prob, acc = train_mlp(X_tr, y_tr, X_te, y_te, class_names, epochs=epochs, batch_size=batch_size)

    # Save outputs next to CSV
    out_dir = os.path.dirname(csv_path) or "."
    out_csv = os.path.join(out_dir, "predictions.csv")

    # For binary: 2 columns; for multiclass: len(class_names)
    prob_cols = [f"conf_{name}" for name in class_names]
    out_df = pd.DataFrame({
        "pred_label": [class_names[i] for i in y_pred],
        "true_label": [class_names[i] for i in y_te],
    })
    for i, cname in enumerate(class_names):
        out_df[prob_cols[i]] = y_prob[:, i]
    out_df.to_csv(out_csv, index=False)

    meta = {
        "binary": binary,
        "model": kind,
        "features": features,
        "classes": class_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "val_accuracy": float(acc),
    }
    with open(os.path.join(out_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nWrote: {out_csv} and model_meta.json")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to Kepler CSV")
    ap.add_argument("--binary", action="store_true", help="Planet vs Not (CONFIRMED/CANDIDATE -> PLANET)")
    ap.add_argument("--model", choices=["tree","mlp"], default="tree", help="Classifier type")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()
    main(args.csv, binary=args.binary, model_choice=args.model, epochs=args.epochs, batch_size=args.batch_size)
