"""Trains SmallCIFARNet on CIFAR-10 for a few epochs and exports it to
ONNX. This is not a research training run: just enough epochs that the
model has real accuracy to lose, which is the precondition for the
accuracy-budget gate in gate.py to mean anything."""
import argparse
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from model.cnn import SmallCIFARNet

DATA_ROOT = "/tmp/cifar10"


def get_loaders(batch_size=128):
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_tf)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--checkpoint", default="model.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SmallCIFARNet().to(device)
    train_loader, test_loader = get_loaders()

    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=args.epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch + 1}/{args.epochs} test_top1={acc:.4f} elapsed={time.time() - t0:.1f}s")

    torch.save(model.state_dict(), args.checkpoint)

    model.eval()
    dummy = torch.randn(1, 3, 32, 32, device=device)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )
    print(f"exported {args.out}")


if __name__ == "__main__":
    main()
