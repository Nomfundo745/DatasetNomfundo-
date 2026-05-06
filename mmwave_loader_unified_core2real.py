# mmwave_loader_unified.py
# Loader that fixes MATLAB-style "sample last" layouts:
#   cnnInput (64,64,1,N) -> (N,1,64,64)
#   cyclo_f  (15,N)      -> (N,15)
# Works for v7 (SciPy) and v7.3 (h5py). Returns PyTorch loaders.

import os
import collections
import numpy as np
import scipy.io as sio
import h5py
import torch
from torch.utils.data import Dataset, DataLoader, random_split

DROP_BAD_ROWS = True  # drop rows with NaN/Inf

# ---------------- HDF5 string helpers ----------------
def _decode_h5_charlike(obj) -> str:
    data = obj[()] if isinstance(obj, h5py.Dataset) else obj
    if isinstance(data, bytes):
        return data.decode("utf-8", errors="ignore").strip("\x00").strip()
    arr = np.array(data)
    if arr.dtype == np.object_:
        try:
            return b"".join(arr.tolist()).decode("utf-8", errors="ignore").strip("\x00").strip()
        except Exception:
            return str(arr)
    if arr.dtype == np.uint16:
        return arr.tobytes().decode("utf-16le", errors="ignore").replace("\x00", "").strip()
    if arr.dtype == np.uint8:
        return arr.tobytes().decode("utf-8", errors="ignore").replace("\x00", "").strip()
    try:
        return arr.astype(np.uint16).tobytes().decode("utf-16le", errors="ignore").replace("\x00", "").strip()
    except Exception:
        return str(arr)

def _read_h5_string_array(f: h5py.File, node) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        try:
            return np.array(node.asstr()[:], dtype=str).reshape(-1)
        except Exception:
            pass
        if node.dtype == np.object_ or node.dtype.kind == "O":
            out = []
            for ref in np.array(node[()]).flat:
                if isinstance(ref, h5py.Reference) and ref:
                    out.append(_decode_h5_charlike(f[ref]))
                else:
                    out.append("unknown")
            return np.array(out, dtype=str).reshape(-1)
        return np.array([_decode_h5_charlike(node)], dtype=str)
    return np.array(["unknown"], dtype=str)

def _as_str_array(x, fallback_len=None):
    if x is None:
        return np.array(["unknown"] * int(fallback_len or 1))
    arr = np.array(x, dtype=object).squeeze()
    if arr.shape == ():
        arr = np.array([arr], dtype=object)
    out = []
    for v in arr.tolist():
        if isinstance(v, (bytes, bytearray)):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            sv = str(v)
            if sv.startswith("b'") and sv.endswith("'"):
                try:
                    sv = sv[2:-1].encode("latin-1").decode("utf-8", errors="ignore")
                except Exception:
                    sv = sv[2:-1]
            out.append(sv)
    return np.array(out, dtype=str)

def _to_np(obj):
    return np.array(obj) if not isinstance(obj, h5py.Dataset) else np.array(obj)

def _read_scalar(ds):
    a = _to_np(ds)
    return a.item() if a.shape == () else a.squeeze()

# ---------------- axis-fix utilities ----------------
def _fix_cnn_shape(arr):
    """
    Convert various MATLAB-ish layouts to (N,1,64,64).
    Accepts:
      (64,64,1,N), (1,64,64,N), (64,64,N), (64,N,64), (N,64,64), (N,1,64,64)
    """
    a = np.array(arr)
    if a.ndim == 2 and a.shape == (64, 64):
        # Single image -> N=1
        a = a[None, None, :, :]
        return a.astype("float32")

    if a.ndim == 3:
        # Possible (64,64,N) or (64,N,64) or (N,64,64)
        if a.shape[0] == 64 and a.shape[1] == 64:
            # (64,64,N) -> (N,1,64,64)
            a = np.transpose(a, (2, 0, 1))[:, None, :, :]
            return a.astype("float32")
        if a.shape[0] == 64 and a.shape[2] == 64:
            # (64,N,64) -> (N,1,64,64)
            a = np.transpose(a, (1, 0, 2))[:, None, :, :]
            return a.astype("float32")
        if a.shape[1] == 64 and a.shape[2] == 64:
            # (N,64,64) -> (N,1,64,64)
            a = a[:, None, :, :]
            return a.astype("float32")

    if a.ndim == 4:
        # Try all permutations to find two 64s and a 1 channel
        Hcand = [i for i,d in enumerate(a.shape) if d == 64]
        ones  = [i for i,d in enumerate(a.shape) if d == 1]
        if len(Hcand) >= 2:
            # pick two positions that are 64
            # try placing them as (H,W)=(64,64); channel=1 if present; remaining axis = N
            for h in Hcand:
                for w in Hcand:
                    if h == w: 
                        continue
                    # find channel axis
                    c = ones[0] if ones else None
                    axes = set(range(4))
                    if c is not None:
                        rest = list(axes - {h, w, c})
                        if len(rest) == 1:
                            n = rest[0]
                            # reorder to (N, C, H, W)
                            out = np.transpose(a, (n, c, h, w))
                            return out.astype("float32")
                    # If no explicit channel=1, accept any axis as channel and slice 1
                    rest = list(axes - {h, w})
                    if len(rest) == 2:
                        n, c = rest  # guess: first is N, second is C
                        out = np.transpose(a, (n, c, h, w))
                        # force single channel
                        if out.shape[1] != 1:
                            out = out[:, :1, :, :]
                        return out.astype("float32")
        # common specific: (64,64,1,N) -> (N,1,64,64)
        if a.shape[:3] == (64,64,1):
            return np.transpose(a, (3,2,0,1)).astype("float32")
        # (1,64,64,N) -> (N,1,64,64)
        if a.shape[:3] == (1,64,64):
            return np.transpose(a, (3,0,1,2)).astype("float32")
        # (N,1,64,64) already
        if a.shape[1:] == (1,64,64):
            return a.astype("float32")

    raise ValueError(f"cnnInput has unsupported shape {a.shape}; expected like (64,64,1,N) or (N,1,64,64).")

def _fix_feat_shape(arr):
    """Make (N,15) from (15,N) or (N,15)."""
    a = np.array(arr)
    if a.ndim == 1 and a.size == 15:
        return a[None, :].astype("float32")
    if a.ndim == 2:
        if a.shape[1] == 15:
            return a.astype("float32")
        if a.shape[0] == 15:
            return a.T.astype("float32")
    raise ValueError(f"cyclo features shape {a.shape} not understood. Expect (N,15) or (15,N).")

def _fix_vec_shape(arr, name="vec"):
    """Make (N,) from (N,1), (1,N), (N,), (1,1,N), (N,1,1) etc."""
    a = np.array(arr).squeeze()
    if a.ndim == 0:
        return a.reshape(1).astype("float32" if name!="y" else "int64")
    if a.ndim == 1:
        return a.astype("float32" if name!="y" else "int64")
    if a.ndim == 2:
        if a.shape[0] == 1:
            return a.reshape(-1).astype("float32" if name!="y" else "int64")
        if a.shape[1] == 1:
            return a.reshape(-1).astype("float32" if name!="y" else "int64")
    if a.ndim == 3 and 1 in a.shape:
        return a.reshape(-1).astype("float32" if name!="y" else "int64")
    return a.reshape(-1).astype("float32" if name!="y" else "int64")

# ---------------- Dataset class ----------------
class MmWaveFusionDataset(Dataset):
    def __init__(self, X_img, X_feat, y, fc, snr, src=None):
        self.X_img = torch.from_numpy(X_img).float()
        self.X_feat = torch.from_numpy(X_feat).float()
        self.y = torch.from_numpy(y).long()
        self.fc = torch.from_numpy(fc).float()
        self.snr = torch.from_numpy(snr).float()

        if src is None:
            self.src = None
            self.src_idx = torch.full((len(self.y),), -1, dtype=torch.long)
            self.src_vocab = {}
        else:
            src = np.array(src).astype(str).reshape(-1)
            uniq = sorted(set(src))
            vocab = {s: i for i, s in enumerate(uniq)}
            self.src_idx = torch.tensor([vocab.get(s, -1) for s in src], dtype=torch.long)
            self.src = src
            self.src_vocab = vocab

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.X_img[idx], self.X_feat[idx], self.y[idx], self.fc[idx], self.snr[idx], self.src_idx[idx]

# ---------------- Readers for v7 / v7.3 ----------------
def _read_flat_v7(mat):
    need = ["cnnInput", "cyclo_f", "labels", "fc", "snr"]
    if all(k in mat for k in need):
        X_img = mat["cnnInput"]
        X_feat = mat["cyclo_f"]
        y = mat["labels"]
        fc = mat["fc"]
        snr = mat["snr"]
        src = mat.get("src", None)
        return X_img, X_feat, y, fc, snr, src
    return None

def _read_struct_v7(mat):
    if "dataset" not in mat: return None
    ds = mat["dataset"]
    items = (ds,) if not isinstance(ds, np.ndarray) or ds.size == 1 else ds.flat
    X_img, X_feat, y, fc, snr, src = [], [], [], [], [], []
    for item in items:
        def _get(field, default=None):
            if hasattr(item, field): return getattr(item, field)
            if isinstance(item, np.void) and item.dtype.names and field in item.dtype.names: return item[field]
            return default
        img=_get("cnnInput"); feat=_get("cyclo"); cls=_get("class")
        fchz=_get("fc"); sndb=_get("SNRdB"); srcs=_get("source", "unknown")
        if img is None or feat is None or cls is None or fchz is None or sndb is None: continue
        X_img.append(np.array(img)); X_feat.append(np.array(feat)); y.append(np.array(cls))
        fc.append(np.array(fchz)); snr.append(np.array(sndb)); src.append(srcs)
    # Stack later after axis fix
    return X_img, X_feat, y, fc, snr, src

def _read_flat_h5(f: h5py.File):
    need = ["cnnInput", "cyclo_f", "labels", "fc", "snr"]
    if all(k in f.keys() for k in need):
        X_img = f["cnnInput"][()]
        X_feat = f["cyclo_f"][()]
        y = f["labels"][()]
        fc = f["fc"][()]
        snr = f["snr"][()]
        src = _read_h5_string_array(f, f["src"]) if "src" in f.keys() else None
        return X_img, X_feat, y, fc, snr, src
    return None

def _read_struct_h5(f: h5py.File):
    if "dataset" not in f: return None
    grp = f["dataset"]

    # Arrays-of-refs case
    cand = ["cnnInput","cyclo","class","fc","SNRdB"]
    if all(k in grp.keys() for k in cand):
        def deref_array(ds):
            arr = np.array(ds, dtype=object); out=[]
            for ref in arr.flat:
                if isinstance(ref, h5py.Reference) and ref: out.append(np.array(f[ref]))
                else: out.append(None)
            return out
        cnn = deref_array(grp["cnnInput"]); cyc = deref_array(grp["cyclo"])
        cls = deref_array(grp["class"]);    fc  = deref_array(grp["fc"])
        snr = deref_array(grp["SNRdB"]);    src = deref_array(grp["source"]) if "source" in grp.keys() else [None]*len(cnn)

        X_img,X_feat,y,fcHz,snrDb,srcs=[],[],[],[],[],[]
        for i in range(len(cnn)):
            if cnn[i] is None: continue
            X_img.append(cnn[i]); X_feat.append(cyc[i]); y.append(cls[i]); fcHz.append(fc[i]); snrDb.append(snr[i])
            # decode string ref to python str (later expand)
            if src[i] is None:
                srcs.append("unknown")
            else:
                try:
                    s_i = _decode_h5_charlike(f[src[i]]) if isinstance(src[i], h5py.Reference) else _decode_h5_charlike(src[i])
                except Exception:
                    s_i = "unknown"
                srcs.append(s_i)
        return X_img, X_feat, y, fcHz, snrDb, np.array(srcs, dtype=str)

    # Per-element subgroups
    elem_keys = [k for k in grp.keys() if isinstance(grp[k], h5py.Group)]
    if not elem_keys: return None
    try: elem_keys.sort(key=lambda x:int(x))
    except: elem_keys.sort()

    X_img,X_feat,y,fcHz,snrDb,srcs=[],[],[],[],[],[]
    for ek in elem_keys:
        eg = grp[ek]
        if not all(k in eg.keys() for k in ["cnnInput","cyclo","class","fc","SNRdB"]): continue
        X_img.append(np.array(eg["cnnInput"]))
        X_feat.append(np.array(eg["cyclo"]))
        y.append(eg["class"][()])
        fcHz.append(eg["fc"][()])
        snrDb.append(eg["SNRdB"][()])
        if "source" in eg.keys():
            try: srcs.append(_read_h5_string_array(f, eg["source"])[0])
            except Exception: srcs.append("unknown")
        else:
            srcs.append("unknown")
    return X_img, X_feat, y, fcHz, snrDb, np.array(srcs, dtype=str)

# ---------------- main loader ----------------
def load_mmwave_dataset(mat_path, batch_size=32, val_ratio=0.2, shuffle_train=True):
    if not (isinstance(mat_path, str) and mat_path.lower().endswith(".mat") and os.path.isfile(mat_path)):
        folder = os.path.dirname(mat_path) if isinstance(mat_path, str) else os.getcwd()
        mats = [f for f in os.listdir(folder) if f.lower().endswith(".mat")] if os.path.isdir(folder) else []
        raise FileNotFoundError(
            f"MAT file not found: {mat_path}\nIn folder: {folder}\nFound .mat files here: {mats}"
        )

    print(f"Loading dataset from: {mat_path}")

    # Try SciPy (v7 or older)
    try:
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        print("✅ Loaded using scipy.io")
        flat = _read_flat_v7(mat)
        if flat is not None:
            X_img, X_feat, y, fc, snr, src = flat
        else:
            X_img, X_feat, y, fc, snr, src = _read_struct_v7(mat)
    except NotImplementedError:
        # v7.3 (HDF5)
        print("⚠️ Detected v7.3 file, using h5py")
        with h5py.File(mat_path, "r") as f:
            flat = _read_flat_h5(f)
            if flat is not None:
                X_img, X_feat, y, fc, snr, src = flat
            else:
                X_img, X_feat, y, fc, snr, src = _read_struct_h5(f)

    # -------- axis fixes --------
    # cnnInput
    X_img = _fix_cnn_shape(X_img)

    # features
    X_feat = _fix_feat_shape(X_feat)

    # labels/fc/snr vectors
    y   = _fix_vec_shape(y,   "y").astype("int64")
    fc  = _fix_vec_shape(fc,  "fc").astype("float32")
    snr = _fix_vec_shape(snr, "snr").astype("float32")

    # src length fix
    if src is None:
        src = np.array(["unknown"] * len(y))
    else:
        src = np.array(src).reshape(-1)
        if src.size == 1 and len(y) > 1:
            src = np.array([src.item()] * len(y))
        elif src.size != len(y):
            # last resort: pad/trim
            if src.size < len(y):
                src = np.concatenate([src, np.array(["unknown"] * (len(y) - src.size))])
            else:
                src = src[:len(y)]

    # -------- trim to common N --------
    lens = [X_img.shape[0], X_feat.shape[0], y.shape[0], fc.shape[0], snr.shape[0], src.shape[0]]
    N = min(lens)
    if len(set(lens)) != 1:
        print(f"⚠️ Lengths before trim (img,feat,y,fc,snr,src) = {tuple(lens)} → trimming to N={N}")
    X_img, X_feat, y, fc, snr, src = X_img[:N], X_feat[:N], y[:N], fc[:N], snr[:N], src[:N]

    # -------- drop NaN/Inf rows --------
    if DROP_BAD_ROWS:
        img_flat = X_img.reshape(N, -1)
        mask = np.isfinite(img_flat).all(axis=1)
        mask &= np.isfinite(X_feat).all(axis=1)
        mask &= np.isfinite(y)
        mask &= np.isfinite(fc)
        mask &= np.isfinite(snr)
        keep = int(mask.sum())
        if keep < N:
            print(f"⚠️ Dropping {N-keep} rows with NaN/Inf; using {keep}.")
            X_img = X_img[mask]; X_feat = X_feat[mask]; y = y[mask]; fc = fc[mask]; snr = snr[mask]; src = src[mask]
            N = keep

    # -------- final report --------
    class_counts = dict(collections.Counter(y.tolist()))
    print("✅ cnnInput shape:", X_img.shape)   # (N,1,64,64)
    print("✅ cyclo_features shape:", X_feat.shape)  # (N,15)
    print("✅ labels shape:", y.shape)
    print("✅ fc shape:", fc.shape)
    print("✅ snr shape:", snr.shape)
    print("Class distribution:", class_counts)

    # dataset & loaders
    ds = MmWaveFusionDataset(X_img, X_feat, y, fc, snr, src)
    val_size = max(1, int(len(ds) * val_ratio))
    train_size = max(0, len(ds) - val_size)
    if train_size == 0:
        print("⚠️ Very few samples after alignment; validation set may be empty.")
        train_size = len(ds); val_size = 0
    tr_ds, va_ds = random_split(ds, [train_size, val_size])
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=shuffle_train) if train_size > 0 else None
    va = DataLoader(va_ds, batch_size=batch_size, shuffle=False) if val_size > 0 else None
    return X_img, X_feat, y, fc, snr, src, tr, va, class_counts

# ---------------- standalone test ----------------
if __name__ == "__main__":
    MAT_PATH = r"C:\Users\NOMFUNDO\Documents\MATLAB\Project2\mmWaveDataset_Balanced_DWPT_fromUnified_FULL.mat"
    X_img, X_feat, y, fc, snr, src, train_loader, val_loader, class_counts = load_mmwave_dataset(
        MAT_PATH, batch_size=32, val_ratio=0.2
    )
    print("✅ Loader test complete")
