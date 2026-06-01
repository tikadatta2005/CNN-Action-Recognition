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
    save_checkpoints=None
):
    metrics = []

    # create checkpoint directory if requested
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for i in range(epoch):

        # train for one epoch
        train_metrics = train_model(
            model=model,
            dataloader=train_dataloader,
            lr=lr
        )

        # training only
        if test_dataloader is None:

            metrics.append({
                "epoch": i + 1,
                **train_metrics
            })

            continue

        # validation/testing
        test_metrics = test_model(
            model=model,
            dataloader=test_dataloader
        )

        # progress logging
        if print_on and (i + 1) % print_on == 0:
            print(
                f"Epoch {i+1}/{epoch} | "
                f"Train Loss: {train_metrics['training_loss']:.4f} | "
                f"Validation Loss: {test_metrics['loss']:.4f}"
            )

        # checkpoint saving
        if (
            save_checkpoints
            and save_dir is not None
            and (i + 1) % save_checkpoints == 0
        ):
            torch.save(
                model.state_dict(),
                save_dir / f"checkpoint_epoch_{i+1}.pth"
            )

        metrics.append({
            "epoch": i + 1,
            **train_metrics,
            **{
                f"test_{k}": v
                for k, v in test_metrics.items()
            }
        })
    #save the final model   
    if save_dir is not None:
        torch.save(
            model.state_dict(),
            save_dir / "final_model.pth"
        )

    return metrics