import argparse, os, json, warnings, sys, joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Canonical Kepler feature list (order matters!)
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

    df = df[df[LABEL_COL].isin(MULTI_CLASSES)].copy()

    if binary:
        y = np.where(df[LABEL_COL].isin(["CONFIRMED", "CANDIDATE"]), 1, 0).astype(int)
        class_names = ["NOT", "PLANET"]
    else:
        class_to_idx = {cls: i for i, cls in enumerate(MULTI_CLASSES)}
        y = df[LABEL_COL].map(class_to_idx).astype(int).values
        class_names = MULTI_CLASSES

    X = df[[c for c in KEPLER_FEATURES if c in df.columns]].astype(np.float32).values
    used_feats = [c for c in KEPLER_FEATURES if c in df.columns]
    return X, y, used_feats, class_names

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

def train_mlp(X_tr, y_tr, X_te, y_te, class_names, epochs=60, batch=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = np.unique(y_tr)
    weights_np = compute_class_weight(class_weight="balanced", classes=classes, y=y_tr).astype(np.float32)
    class_weights = torch.from_numpy(weights_np).to(device=device, dtype=torch.float32)

    train_loader = DataLoader(TabularDS(X_tr, y_tr), batch_size=batch, shuffle=True)
    test_loader  = DataLoader(TabularDS(X_te, y_te), batch_size=max(256, batch), shuffle=False)

    model = MLP(in_dim=X_tr.shape[1], out_dim=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=4, factor=0.5)

    best_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
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
        preds = []
        with torch.no_grad():
            for xb, _ in test_loader:
                logits = model(xb.to(device))
                preds.append(torch.softmax(logits, dim=1).cpu().numpy())
        y_prob = np.vstack(preds)
        y_pred = y_prob.argmax(1)
        acc = accuracy_score(y_te, y_pred)
        scheduler.step(acc)
        tqdm.write(f"Epoch {epoch:02d} | val_acc={acc:.4f}")

        if acc > best_acc + 1e-4:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict({k: v for k, v in best_state.items()})

    return model

def eval_and_print(y_te, y_pred, y_prob, class_names, tag="Model"):
    print(f"\n[{tag}] Test accuracy:", accuracy_score(y_te, y_pred))
    print(f"\n[{tag}] Classification report:")
    print(classification_report(y_te, y_pred, target_names=class_names))
    print(f"\n[{tag}] Confusion matrix:\n", confusion_matrix(y_te, y_pred))

# ---------- Train command ----------
def cmd_train(args):
    print("numpy:", np.__version__)
    df = load_csv_smart(args.csv)
    X, y, features, class_names = prepare_kepler(df, binary=args.binary)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.21, random_state=SEED, stratify=y)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    os.makedirs(args.out, exist_ok=True)
    meta = {
        "binary": args.binary,
        "model": args.model,
        "features": features,
        "classes": class_names,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }

    if args.model == "tree":
        model = HistGradientBoostingClassifier(
            learning_rate=0.06, max_depth=8, max_iter=800,
            l2_regularization=0.0, min_samples_leaf=20, random_state=SEED
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        y_prob = model.predict_proba(X_te)
        eval_and_print(y_te, y_pred, y_prob, class_names, tag="Tree")
        joblib.dump(model, os.path.join(args.out, "model.joblib"))
        np.savez(os.path.join(args.out, "scaler.npz"), mean=scaler.mean_, scale=scaler.scale_)
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nSaved artifacts to: {args.out}")
    else:
        # MLP path
        model = train_mlp(X_tr, y_tr, X_te, y_te, class_names, epochs=args.epochs, batch=args.batch_size)
        # Final eval
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            y_prob = torch.softmax(torch.tensor(model(np.asarray(X_te, dtype=np.float32))).to("cpu"), dim=1).numpy()
        y_pred = y_prob.argmax(1)
        eval_and_print(y_te, y_pred, y_prob, class_names, tag="MLP")
        # Save artifacts
        torch.save({"state_dict": model.state_dict(),
                    "in_dim": X_tr.shape[1],
                    "out_dim": len(class_names)}, os.path.join(args.out, "model.pt"))
        np.savez(os.path.join(args.out, "scaler.npz"), mean=scaler.mean_, scale=scaler.scale_)
        with open(os.path.join(args.out, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\nSaved artifacts to: {args.out}")

# ---------- Predict utilities ----------
def load_artifacts(art_dir):
    with open(os.path.join(art_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    sc = np.load(os.path.join(art_dir, "scaler.npz"))
    scaler_mean, scaler_scale = sc["mean"], sc["scale"]
    scaler = StandardScaler()
    # Manually set trained params
    scaler.mean_ = scaler_mean
    scaler.scale_ = scaler_scale

    if meta["model"] == "tree":
        model = joblib.load(os.path.join(art_dir, "model.joblib"))
        model_type = "tree"
    else:
        # Rebuild and load MLP
        in_dim = len(meta["features"])
        out_dim = len(meta["classes"])
        model = MLP(in_dim, out_dim)
        state = torch.load(os.path.join(art_dir, "model.pt"), map_location="cpu")
        model.load_state_dict(state["state_dict"])
        model.eval()
        model_type = "mlp"
    return meta, scaler, model, model_type

def vector_from_kwargs(meta, **kwargs):
    feats = meta["features"]
    x = []
    for f in feats:
        if f not in kwargs or kwargs[f] is None:
            raise ValueError(f"Missing required feature: {f}")
        x.append(float(kwargs[f]))
    return np.array(x, dtype=np.float32).reshape(1, -1)

def predict_single(art_dir, **feature_kwargs):
    meta, scaler, model, model_type = load_artifacts(art_dir)
    x = vector_from_kwargs(meta, **feature_kwargs)
    xs = (x - scaler.mean_) / scaler.scale_

    if model_type == "tree":
        probs = model.predict_proba(xs)[0]
    else:
        with torch.no_grad():
            probs = torch.softmax(torch.tensor(model(xs.astype(np.float32))), dim=1).numpy()[0]

    idx = int(np.argmax(probs))
    label = meta["classes"][idx]
    return label, {meta["classes"][i]: float(probs[i]) for i in range(len(probs))}

def predict_csv(art_dir, csv_path, out_path=None):
    meta, scaler, model, model_type = load_artifacts(art_dir)
    df = load_csv_smart(csv_path)

    # Prepare features in correct order; coerce/median-impute
    feats = meta["features"]
    for c in feats:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
        if df[c].isna().all():
            df[c] = 0.0
        else:
            df[c] = df[c].fillna(df[c].median())

    X = df[feats].astype(np.float32).values
    Xs = (X - scaler.mean_) / scaler.scale_

    if model_type == "tree":
        y_prob = model.predict_proba(Xs)
    else:
        with torch.no_grad():
            y_prob = torch.softmax(torch.tensor(model(Xs.astype(np.float32))), dim=1).numpy()

    y_pred = y_prob.argmax(1)
    out = pd.DataFrame({"pred_label": [meta["classes"][i] for i in y_pred]})
    for i, cname in enumerate(meta["classes"]):
        out[f"conf_{cname}"] = y_prob[:, i]

    if out_path:
        out.to_csv(out_path, index=False)
        print(f"Wrote predictions to {out_path}")
    return out

# ---------- Predict command ----------
def cmd_predict(args):
    if args.from_csv:
        predict_csv(args.artifacts, args.from_csv, out_path=args.out)
    else:
        # Build kwargs from provided CLI flags matching feature names
        with open(os.path.join(args.artifacts, "meta.json"), "r") as f:
            meta = json.load(f)
        feats = meta["features"]

        kwargs = {}
        for f in feats:
            # argparse stores dashes as underscore; our features have underscores already
            val = getattr(args, f, None)
            kwargs[f] = val
        label, probs = predict_single(args.artifacts, **kwargs)
        print("\nPREDICTION:", label)
        print("PROBS:", json.dumps(probs, indent=2))

# ---------- Print-schema command ----------
def cmd_print_schema(args):
    with open(os.path.join(args.artifacts, "meta.json"), "r") as f:
        meta = json.load(f)
    print("Required features, in order:")
    for f in meta["features"]:
        print("-", f)
    print("\nClasses:", meta["classes"])
    print("Binary mode:", meta["binary"])
    print("Model:", meta["model"])

# ---------- CLI ----------
def build_parser():
    ap = argparse.ArgumentParser(description="Kepler ML CLI: train/export, load, and predict from CLI")
    sp = ap.add_subparsers(dest="cmd", required=True)

    tr = sp.add_parser("train", help="Train and export artifacts")
    tr.add_argument("--csv", required=True, help="Path to Kepler CSV")
    tr.add_argument("--binary", action="store_true", help="Planet vs Not (CONFIRMED/CANDIDATE -> PLANET)")
    tr.add_argument("--model", choices=["tree","mlp"], default="tree")
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--out", default="./artifacts")
    tr.set_defaults(func=cmd_train)

    pr = sp.add_parser("predict", help="Predict from CLI flags or CSV")
    pr.add_argument("--artifacts", required=True)
    pr.add_argument("--from-csv", help="Batch predict a CSV")
    pr.add_argument("--out", help="Where to write CSV predictions")
    # Dynamically add feature flags so you can do --koi_period 9.49 etc.
    for feat in KEPLER_FEATURES:
        pr.add_argument(f"--{feat}", type=float, required=False, help=f"Feature: {feat}")
    pr.set_defaults(func=cmd_predict)

    sc = sp.add_parser("print-schema", help="Print required features and classes for this artifact")
    sc.add_argument("--artifacts", required=True)
    sc.set_defaults(func=cmd_print_schema)

    return ap

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
