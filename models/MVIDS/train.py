import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay, confusion_matrix
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

def train_loop(model: nn.Module, dl_tr, dl_va, args, kg_vec_fn=None, save_path: str = None):
    device = args.device
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='max',
        factor=0.5,
        patience=3,
    )
    bce = nn.BCEWithLogitsLoss()

    best_f1, best_state = 0.0, None
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        train_bar = tqdm(dl_tr, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)
        for xb_views, yb in train_bar:
            xb_views = [x.to(device) for x in xb_views]
            yb = yb.to(device)
            opt.zero_grad()
            if kg_vec_fn is not None:
                kg = kg_vec_fn(xb_views)
                kg = kg.to(device) if hasattr(kg, "to") else kg
                logits = model(xb_views, kg)
            else:
                logits = model(xb_views)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            train_bar.set_postfix(loss=np.mean(losses[-10:]) if losses else 0.0)
        
        # validation이 있을 때만 실행
        if dl_va is not None:
            model.eval()
            all_prob, all_true = [], []
            with torch.no_grad():
                val_bar = tqdm(dl_va, desc=f"Epoch {epoch}/{args.epochs} [val]", leave=False)
                for xb_views, yb in val_bar:
                    xb_views = [x.to(device) for x in xb_views]
                    if kg_vec_fn is not None:
                        kg = kg_vec_fn(xb_views)
                        kg = kg.to(device) if hasattr(kg, "to") else kg
                        logits = model(xb_views, kg)
                    else:
                        logits = model(xb_views)
                    prob = torch.sigmoid(logits).cpu().numpy()
                    all_prob.append(prob)
                    all_true.append(yb.numpy())
            all_prob = np.concatenate(all_prob)
            all_true = np.concatenate(all_true)
            m = compute_metrics(all_true, all_prob)
            sched.step(m['f1'])
            print(f"Epoch {epoch:03d}/{args.epochs} | TrainLoss={np.mean(losses):.4f} | ValF1={m['f1']:.4f} Acc={m['acc']:.4f}")
            if m['f1'] > best_f1:
                best_f1 = m['f1']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            # validation 없이 train loss만 출력
            print(f"Epoch {epoch:03d}/{args.epochs} | TrainLoss={np.mean(losses):.4f}")
            # validation 없이는 마지막 epoch의 state를 저장
            if epoch == args.epochs:
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        # 모델 저장
        if save_path is not None:
            torch.save(best_state, save_path)
            print(f"Best model saved to {save_path}")
    return model


def test_loop(model: nn.Module, dl_te, args, kg_vec_fn=None):
    device = args.device
    model.to(device)
    model.eval()
    all_prob, all_true = [], []
    with torch.no_grad():
        test_bar = tqdm(dl_te, desc="Test", leave=False)
        for xb_views, yb in test_bar:
            xb_views = [x.to(device) for x in xb_views]
            if kg_vec_fn is not None:
                kg = kg_vec_fn(xb_views)
                kg = kg.to(device) if hasattr(kg, "to") else kg
                logits = model(xb_views, kg)
            else:
                logits = model(xb_views)
            prob = torch.sigmoid(logits).cpu().numpy()
            all_prob.append(prob)
            all_true.append(yb.numpy())
    all_prob = np.concatenate(all_prob)
    all_true = np.concatenate(all_true)
    m = compute_metrics(all_true, all_prob)
    print(f"Test: F1={m['f1']:.4f} Acc={m['acc']:.4f} Pre={m['pre']:.4f} Rec={m['rec']:.4f}")

    # Confusion matrix plot
    y_pred = (all_prob >= 0.5).astype(int)
    cm = confusion_matrix(all_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    img_dir = Path("img")
    img_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    return m

def compute_metrics(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    return dict(
        f1=f1_score(y_true, y_pred, zero_division=0),
        acc=accuracy_score(y_true, y_pred),
        pre=precision_score(y_true, y_pred, zero_division=0),
        rec=recall_score(y_true, y_pred, zero_division=0),
    )
