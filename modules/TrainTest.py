import torch
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# --------------------------------------------------
# Train model for one epoch
# --------------------------------------------------
def train_model(model, dataloader, lr=0.001):
    model.train()

    total_loss = 0

    all_y_true = []
    all_y_pred = []

    for x, y in dataloader:
        # move batch to GPU
        x = x.to("cuda", non_blocking=True)
        y = y.to("cuda", non_blocking=True)

        # forward pass
        logits = model(x)

        # compute loss
        loss = model.calculate_loss(logits, y)

        # update weights
        loss = model.backward(loss, lr)

        total_loss += loss

        # predicted class index
        preds = torch.argmax(logits, dim=1)

        all_y_true.extend(y.cpu().numpy())
        all_y_pred.extend(preds.cpu().numpy())

    return {
        "training_loss": total_loss / len(dataloader),
        "training_accuracy": accuracy_score(all_y_true, all_y_pred),
        "training_precision": precision_score(
            all_y_true,
            all_y_pred,
            average="macro",
            zero_division=0
        ),
        "training_recall": recall_score(
            all_y_true,
            all_y_pred,
            average="macro",
            zero_division=0
        ),
        "training_f1": f1_score(
            all_y_true,
            all_y_pred,
            average="macro",
            zero_division=0
        ),
    }


# --------------------------------------------------
# Evaluate model (validation / test)
# --------------------------------------------------
def test_model(model, dataloader):
    model.eval()

    total_loss = 0

    all_y_true = []
    all_y_pred = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to("cuda", non_blocking=True)
            y = y.to("cuda", non_blocking=True)

            logits = model(x)

            loss = model.calculate_loss(logits, y)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_y_true.extend(y.cpu().numpy())
            all_y_pred.extend(preds.cpu().numpy())

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": accuracy_score(all_y_true, all_y_pred),
        "precision": precision_score(
            all_y_true,
            all_y_pred,
            average="macro",
            zero_division=0
        ),
        "recall": recall_score(
            all_y_true,
            all_y_pred,
            average="macro",
            zero_division=0
        ),
        "f1": f1_score(
            all_y_true,
            all_y_pred,
            average="macro",
            zero_division=0
        ),
    }


# --------------------------------------------------
# Training loop
# --------------------------------------------------
def trainer(
    model,
    train_dataloader,
    test_dataloader=None,
    epoch=1,
    lr=0.001,
    print_on=10,
    save_dir=None,
    save_checkpoints=None,
    checkpoint_name="checkpoint",

    early_stop_patience=None,
    monitor="test_loss",
    min_delta=0.0
):
    metrics = []

    # ----------------------------
    # Setup save directory
    # ----------------------------
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # Early stopping variables
    # ----------------------------
    best_score = None
    bad_epochs = 0

    # ----------------------------
    # Training loop
    # ----------------------------
    for i in range(epoch):

        # ---- train one epoch ----
        train_metrics = train_model(
            model=model,
            dataloader=train_dataloader,
            lr=lr
        )

        # ---- if no validation ----
        if test_dataloader is None:
            metrics.append({"epoch": i + 1, **train_metrics})

            # still allow checkpointing/logging below
            test_metrics = None

        else:
            # ---- validation ----
            test_metrics = test_model(
                model=model,
                dataloader=test_dataloader
            )

            current_score = test_metrics.get(monitor)

            if current_score is None:
                raise ValueError(f"Monitor metric '{monitor}' not found in test_metrics")

            # ----------------------------
            # Early stopping logic
            # ----------------------------
            if early_stop_patience is not None:

                # first epoch initialization
                if best_score is None:
                    best_score = current_score
                    bad_epochs = 0
                    improved = True

                else:
                    improved = current_score < (best_score - min_delta)

                    if improved:
                        best_score = current_score
                        bad_epochs = 0
                    else:
                        bad_epochs += 1

                if bad_epochs >= early_stop_patience:
                    print(
                        f"Early stopping triggered at epoch {i+1}. "
                        f"No improvement for {early_stop_patience} epochs."
                    )
                    break

            # store metrics
            metrics.append({
                "epoch": i + 1,
                **train_metrics,
                **{f"test_{k}": v for k, v in test_metrics.items()}
            })

        # ----------------------------
        # Logging
        # ----------------------------
        if print_on and (i + 1) % print_on == 0:
            if test_dataloader is None:
                print(
                    f"Epoch {i+1}/{epoch} | "
                    f"Train Loss: {train_metrics['training_loss']:.4f}"
                )
            else:
                print(
                    f"Epoch {i+1}/{epoch} | "
                    f"Train Loss: {train_metrics['training_loss']:.4f} | "
                    f"Validation Loss: {test_metrics['loss']:.4f}"
                )

        # ----------------------------
        # Checkpoint saving
        # ----------------------------
        if (
            save_checkpoints
            and save_dir is not None
            and (i + 1) % save_checkpoints == 0
        ):
            torch.save(
                model.state_dict(),
                save_dir / f"{checkpoint_name}_{i+1}.pth"
            )

    # ----------------------------
    # Final model save
    # ----------------------------
    if save_dir is not None:
        torch.save(
            model.state_dict(),
            save_dir / f"final_{checkpoint_name}.pth"
        )

    return metrics