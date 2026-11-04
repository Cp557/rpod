"""Train the M11 keypoint heatmap network and export it to ONNX.

python train.py DATASET OUTPUT_DIR [--epochs N]

DATASET holds `train/` and `val/` from `rpod keypoint-dataset`: 1024x1024 RGB renders and
labels.jsonl with keypoint pixels and visibility. The network is a ResNet-18 encoder with
three 128-channel transposed-convolution stages (stride 4 heatmaps), a narrowed "simple
baseline" pose architecture. Labels use the inference convention of `rpod.keypoints`: heatmap pixel h sits
at network-input pixel 4h + 1.5, and the 512 px input is the 1024 px image area-resampled.
"""

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision

INPUT, STRIDE, OFFSET, SIGMA = 512, 4, 1.5, 1.5
HEATMAP = INPUT // STRIDE
WIDTH = 128
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def network_pixels(full_pixels, full_size):
    """1024 px image pixel centres to 512 px network-input pixel centres."""
    scale = full_size / INPUT
    return (np.asarray(full_pixels) + 0.5) / scale - 0.5


class Renders(Dataset):
    def __init__(self, folder, augment):
        self.folder, self.augment = Path(folder), augment
        self.labels = [json.loads(line) for line in (self.folder / "labels.jsonl").open()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        image = cv2.cvtColor(cv2.imread(str(self.folder / label["image"])), cv2.COLOR_BGR2RGB)
        points = np.asarray(label["keypoints_px"], np.float64)
        visible = np.asarray(label["visible"], bool)
        rng = np.random.default_rng() if self.augment else None
        full_size = image.shape[1]
        if self.augment:
            blur = rng.uniform(0, 0.6)
            if blur > 0.05:
                image = cv2.GaussianBlur(image, (0, 0), sigmaX=blur)
        image = cv2.resize(image.astype(np.float32), (INPUT, INPUT), interpolation=cv2.INTER_AREA)
        points = network_pixels(points, full_size)
        if self.augment:
            # Full-resolution pixel noise up to 2.5 DN, as it is after area resampling.
            noise = rng.uniform(0, 2.5) * INPUT / full_size
            image += noise * rng.standard_normal(image.shape, dtype=np.float32)
            scale, angle = rng.uniform(0.85, 1.15), 0.0
            shift = rng.uniform(-0.08, 0.08, 2) * INPUT
            matrix = cv2.getRotationMatrix2D((INPUT / 2 - 0.5, INPUT / 2 - 0.5), angle, scale)
            matrix[:, 2] += shift
            image = cv2.warpAffine(image, matrix, (INPUT, INPUT), flags=cv2.INTER_LINEAR)
            points = points @ matrix[:, :2].T + matrix[:, 2]
            gain, gamma = rng.uniform(0.6, 1.4), rng.uniform(0.7, 1.4)
            image = 255 * np.clip(image * gain / 255, 0, 1) ** gamma
        inside = (points >= 0).all(axis=1) & (points <= INPUT - 1).all(axis=1)
        visible &= inside
        image = (image / 255 - MEAN) / STD
        grid = np.arange(HEATMAP, dtype=np.float32)
        heat = (points - OFFSET) / STRIDE
        heatmaps = np.zeros((len(points), HEATMAP, HEATMAP), np.float32)
        for k in np.flatnonzero(visible):
            gx = np.exp(-((grid - heat[k, 0]) ** 2) / (2 * SIGMA**2))
            gy = np.exp(-((grid - heat[k, 1]) ** 2) / (2 * SIGMA**2))
            heatmaps[k] = gy[:, None] * gx[None, :]
        return (torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32)),
                torch.from_numpy(heatmaps))


class KeypointNet(nn.Module):
    def __init__(self, keypoints):
        super().__init__()
        backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        layers, channels = [], 512
        for _ in range(3):
            layers += [nn.ConvTranspose2d(channels, WIDTH, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(WIDTH), nn.ReLU(inplace=True)]
            channels = WIDTH
        self.decoder = nn.Sequential(*layers, nn.Conv2d(WIDTH, keypoints, 1))

    def forward(self, image):
        return self.decoder(self.encoder(image))


def pixel_error(predicted, target):
    """Median peak distance in network-input pixels over keypoints labelled visible."""
    count = predicted.shape[1]
    flat_p = predicted.flatten(2).argmax(-1)
    flat_t = target.flatten(2).argmax(-1)
    visible = target.flatten(2).amax(-1) > 0.5
    dp = torch.stack([flat_p % HEATMAP, flat_p // HEATMAP], -1).float()
    dt = torch.stack([flat_t % HEATMAP, flat_t // HEATMAP], -1).float()
    distance = (dp - dt).norm(dim=-1) * STRIDE
    return distance[visible], count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    train = Renders(args.dataset / "train", augment=True)
    val = Renders(args.dataset / "val", augment=False)
    keypoints = len(train.labels[0]["visible"])
    loader = DataLoader(train, args.batch, shuffle=True, num_workers=args.workers,
                        persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val, args.batch, num_workers=args.workers, persistent_workers=True)
    device = torch.device("cuda")
    model = KeypointNet(keypoints).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, epochs=args.epochs,
                                                   steps_per_epoch=len(loader))
    history, best, started = [], math.inf, time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            predicted = model(images)
            weight = 1 + 9 * (targets > 0.1)
            loss = (weight * (predicted - targets) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            schedule.step()
            losses.append(loss.item())
        model.eval()
        errors = []
        with torch.no_grad():
            for images, targets in val_loader:
                distance, _ = pixel_error(model(images.to(device)), targets.to(device))
                errors.append(distance.cpu())
        errors = torch.cat(errors)
        row = {"epoch": epoch, "loss": float(np.mean(losses)),
               "val_median_px": float(errors.median()), "val_p90_px": float(errors.quantile(0.9)),
               "minutes": (time.perf_counter() - started) / 60}
        history.append(row)
        print(json.dumps(row), flush=True)
        if row["val_median_px"] < best:
            best = row["val_median_px"]
            torch.save(model.state_dict(), args.output / "best.pt")
    model.load_state_dict(torch.load(args.output / "best.pt", map_location=device))
    model.eval().cpu()
    torch.onnx.export(model, torch.zeros(1, 3, INPUT, INPUT), args.output / "keypoints.onnx",
                      opset_version=17, input_names=["image"], output_names=["heatmaps"],
                      dynamo=False)
    (args.output / "training.json").write_text(json.dumps({
        "architecture": f"ResNet-18 encoder (ImageNet weights), 3 transposed-conv stages of "
                        f"{WIDTH} channels, stride-4 heatmaps", "input": INPUT, "stride": STRIDE,
        "sigma_heatmap_px": SIGMA, "keypoints": keypoints, "epochs": args.epochs,
        "batch": args.batch, "train_images": len(train), "val_images": len(val),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
        "history": history}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
