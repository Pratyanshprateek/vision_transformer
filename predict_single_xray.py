import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTConfig, ViTForImageClassification


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def choose_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ViTConfig.from_dict(checkpoint["config"])
    model = ViTForImageClassification(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    class_names = checkpoint.get("class_names", ["NORMAL", "PNEUMONIA"])
    return model, class_names


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def predict_image(model, class_names, image_path: Path, device: torch.device):
    image_size = model.config.image_size
    preprocess = build_transform(image_size)
    image = Image.open(image_path).convert("L")
    tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(pixel_values=tensor).logits
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred_idx = int(probabilities.argmax().item())
    return {
        "label": class_names[pred_idx],
        "confidence": float(probabilities[pred_idx].item()),
        "probabilities": {
            class_name: float(probabilities[idx].item())
            for idx, class_name in enumerate(class_names)
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Predict NORMAL vs PNEUMONIA for one chest X-ray image.")
    parser.add_argument("image", type=Path, help="Path to the X-ray image")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/vit_chest_xray_classifier.pt"),
        help="Path to the saved checkpoint",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference even if CUDA or MPS is available",
    )
    args = parser.parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = choose_device(force_cpu=args.cpu)
    model, class_names = load_checkpoint(args.checkpoint, device)
    prediction = predict_image(model, class_names, args.image, device)

    print(f"Device: {device}")
    print(f"Image: {args.image.resolve()}")
    print(f"Predicted class: {prediction['label']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print("Class probabilities:")
    for class_name, probability in prediction["probabilities"].items():
        print(f"  {class_name}: {probability:.4f}")


if __name__ == "__main__":
    main()
