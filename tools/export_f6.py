#!/usr/bin/env python3
"""Strict, fixed-shape F6 exporter: original PyTorch -> NHWC TF -> LiteRT.

This is deliberately NOT a general PyTorch converter. Unsupported graph nodes,
operators, checkpoint differences, and parity failures abort without a manifest.
TensorFlow/TIMM are build-time dependencies only, never server dependencies.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import operator
import os
from pathlib import Path
import shutil
import sys
import tempfile

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
import cv2
import numpy as np
import torch
from torch import nn
from torch.fx import GraphModule, Tracer
from torch.fx.node import map_arg
from export_head import head_bytes


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preprocess(rgb: np.ndarray) -> np.ndarray:
    """Matches source fit_pad with augmentation disabled; result normalized NHWC."""
    h, w = rgb.shape[:2]
    scale = min(384 / w, 800 / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.empty((800, 384, 3), dtype=np.uint8)
    canvas[:] = (124, 116, 104)
    x, y = (384 - nw) // 2, (800 - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    value = canvas.astype(np.float32) / np.float32(255)
    return ((value - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32))[None]


def synthetic_inputs() -> list[np.ndarray]:
    result = []
    for h, w in [(1920, 1080), (1080, 1920), (800, 384), (2400, 1080), (512, 256), (701, 333)]:
        y, x = np.indices((h, w), dtype=np.int32)
        image = np.stack(((x // 7 + y // 3) % 256, (x // 13 * 29 + y // 37 * 43) % 256, (x * 3 + y * 7) % 256), axis=-1).astype(np.uint8)
        # Synthetic UI-like bands/cards, no real screenshot or personal information.
        image[h // 5:h // 3, w // 6:5 * w // 6] = (230, 230, 230)
        result.append(preprocess(image))
    result.extend([preprocess(np.full((800, 384, 3), v, dtype=np.uint8)) for v in (0, 124, 255)])
    return result


class Features(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.forward_features(x).mean((2, 3))


class BackboneTracer(Tracer):
    def is_leaf_module(self, module: nn.Module, qualified_name: str) -> bool:
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.ReLU, nn.ReLU6, nn.Identity, nn.Dropout)):
            return True
        if type(module).__name__ in ("LayerScale2d", "DropPath"):
            return True
        return super().is_leaf_module(module, qualified_name)


class NHWCTranslator:
    def __init__(self, graph: GraphModule, tf):
        self.graph = graph
        self.tf = tf
        self.modules = dict(graph.named_modules())

    def const(self, tensor):
        return self.tf.constant(tensor.detach().cpu().numpy().astype(np.float32))

    def activation(self, module, value):
        tf = self.tf
        if isinstance(module, nn.ReLU6):
            return tf.nn.relu6(value)
        if isinstance(module, nn.ReLU):
            return tf.nn.relu(value)
        if isinstance(module, (nn.Identity, nn.Dropout)) or type(module).__name__ == "DropPath":
            if module.training:
                raise ValueError("Non-eval dropout/identity encountered")
            return value
        raise NotImplementedError(f"Unsupported activation: {type(module)}")

    def layer(self, module: nn.Module, value):
        tf = self.tf
        if isinstance(module, nn.Conv2d):
            if module.padding_mode != "zeros":
                raise NotImplementedError("Only zero-padded convolutions are supported")
            if type(module).__name__ == "Conv2dSame":
                padding = "SAME"
            elif isinstance(module.padding, str):
                if module.padding not in ("valid", "same"):
                    raise NotImplementedError(module.padding)
                padding = module.padding.upper()
            else:
                ph, pw = module.padding
                # PyTorch symmetric padding != TensorFlow SAME for even, stride-2 inputs.
                value = tf.pad(value, [[0, 0], [ph, ph], [pw, pw], [0, 0]]) if ph or pw else value
                padding = "VALID"
            weights = module.weight.detach().cpu().numpy().astype(np.float32)
            sh, sw = module.stride
            dh, dw = module.dilation
            if module.groups == 1:
                kernel = tf.constant(weights.transpose(2, 3, 1, 0))
                output = tf.nn.conv2d(value, kernel, strides=[1, sh, sw, 1], padding=padding, dilations=[1, dh, dw, 1])
            elif module.groups == module.in_channels and module.out_channels == module.in_channels:
                kernel = tf.constant(weights.transpose(2, 3, 0, 1))
                output = tf.nn.depthwise_conv2d(value, kernel, strides=[1, sh, sw, 1], padding=padding, dilations=[dh, dw])
            else:
                raise NotImplementedError("Unexpected grouped convolution in F6")
            return output if module.bias is None else tf.nn.bias_add(output, self.const(module.bias))
        if isinstance(module, nn.BatchNorm2d):
            if module.training or not module.track_running_stats:
                raise ValueError("BatchNorm must use trained running statistics")
            # Compute affine constants first, allowing the converter to fold BN into Conv.
            scale = module.weight.detach() / torch.sqrt(module.running_var.detach() + module.eps) if module.affine else torch.rsqrt(module.running_var.detach() + module.eps)
            bias = module.bias.detach() if module.affine else torch.zeros_like(scale)
            shift = bias - module.running_mean.detach() * scale
            output = value * self.const(scale) + self.const(shift)
            # TIMM BatchNormAct2d contains activation; ignoring it changes the model.
            if hasattr(module, "drop"):
                output = self.activation(module.drop, output)
            return self.activation(module.act, output) if hasattr(module, "act") else output
        if type(module).__name__ == "LayerScale2d":
            return value * tf.reshape(self.const(module.gamma), [1, 1, 1, -1])
        return self.activation(module, value)

    def __call__(self, x):
        values = {}
        for node in self.graph.graph.nodes:
            args = map_arg(node.args, lambda n: values[n])
            kwargs = map_arg(node.kwargs, lambda n: values[n])
            if node.op == "placeholder":
                values[node] = x
            elif node.op == "call_module":
                if len(args) != 1 or kwargs:
                    raise NotImplementedError(f"Unexpected module signature: {node}")
                values[node] = self.layer(self.modules[node.target], args[0])
            elif node.op == "call_function" and node.target in (operator.add, torch.add):
                if kwargs:
                    raise NotImplementedError(node)
                values[node] = args[0] + args[1]
            elif node.op == "call_function" and node.target in (operator.mul, torch.mul):
                if kwargs:
                    raise NotImplementedError(node)
                values[node] = args[0] * args[1]
            elif node.op == "call_method" and node.target == "mean":
                dims = args[1] if len(args) > 1 else kwargs.get("dim")
                keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
                if tuple(dims) != (2, 3) or keep:
                    raise NotImplementedError(f"Unexpected pooling: {node}")
                height, width = int(args[0].shape[1]), int(args[0].shape[2])
                pooled = self.tf.nn.avg_pool2d(args[0], ksize=[height, width], strides=[1, 1], padding="VALID")
                values[node] = self.tf.reshape(pooled, [1, 960])
            elif node.op == "output":
                return args[0]
            else:
                raise NotImplementedError(f"Unsupported graph node: {node.op} {node.target}")
        raise ValueError("Graph did not return features")


def compare(expected: np.ndarray, actual: np.ndarray, label: str, cosine_min: float, max_abs: float) -> dict:
    expected, actual = expected.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    if not np.isfinite(actual).all() or expected.shape != actual.shape:
        raise ValueError(f"{label}: invalid output")
    denominator = np.linalg.norm(expected) * np.linalg.norm(actual)
    cosine = float(np.dot(expected, actual) / max(denominator, 1e-30))
    error = float(np.max(np.abs(expected - actual)))
    if cosine < cosine_min or error > max_abs:
        raise ValueError(f"{label} parity failed: cosine={cosine}, maximum absolute error={error}")
    return {"name": label, "cosine": cosine, "max_abs_error": error}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=root / "models/f6_weights.pt")
    parser.add_argument("--output", type=Path, default=root / "android/app/src/main/assets")
    parser.add_argument("--parity-images", type=Path, help="Optional local PNG/JPEG inputs; never copied into assets")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # A failed new conversion must not leave an old success manifest behind.
    (args.output / "f6_manifest.json").unlink(missing_ok=True)
    import tensorflow as tf
    import timm
    sys.path.insert(0, str(root / "vendor"))
    from ui_foundation.config import experiment_config_from_dict
    from ui_foundation.models.distillation import MobileNetStudent
    torch.set_num_threads(max(1, min(os.cpu_count() or 1, 8)))
    payload = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    provenance = json.loads((root / "models/provenance.json").read_text())
    if sha256(args.checkpoint) != provenance["weights_sha256"]:
        raise ValueError("Checkpoint hash differs from the supplied F6; update/review provenance explicitly")
    config = experiment_config_from_dict(payload["config"])
    config.model.pretrained = False
    config.model.checkpoint = ""
    student = MobileNetStudent(config, output_dim=384).eval()
    student.load_state_dict(payload["student_model_state_dict"], strict=True)
    if student.backbone.feature_dim != 960:
        raise ValueError("F6 trained feature dimension is 960, not config.feature_dim=1280")
    wrapper = Features(student.backbone.model).eval()
    graph = GraphModule(wrapper, BackboneTracer().trace(wrapper)).eval()
    translator = NHWCTranslator(graph, tf)

    class Deployment(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 800, 384, 3], tf.float32, name="normalized_rgb")], autograph=False)
        def encode(self, image):
            return translator(image)

    deployment = Deployment()
    concrete = deployment.encode.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], deployment)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.optimizations = []  # FP32 trained weights; no quantization or retraining.
    model_bytes = converter.convert()
    lite = tf.lite.Interpreter(model_content=model_bytes, num_threads=1, experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    lite.allocate_tensors()
    ops = sorted({entry["op_name"] for entry in lite._get_ops_details()})
    allowed = {"CONV_2D", "DEPTHWISE_CONV_2D", "ADD", "MUL", "SUB", "PAD", "AVERAGE_POOL_2D", "RESHAPE", "RELU", "RELU6"}
    if set(ops) - allowed:
        raise ValueError(f"Graph contains operations outside reviewed GPU set: {set(ops) - allowed}")
    input_spec, output_spec = lite.get_input_details()[0], lite.get_output_details()[0]
    if list(input_spec["shape"]) != [1, 800, 384, 3] or list(output_spec["shape"]) != [1, 960]:
        raise ValueError("Converted tensor shape mismatch")
    inputs = synthetic_inputs()
    if args.parity_images is not None:
        paths = sorted(p for p in args.parity_images.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))[:32]
        if not paths:
            raise ValueError("--parity-images contained no PNG/JPEG images")
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Cannot decode parity image: {path}")
            inputs.append(preprocess(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    checks = []
    golden = None
    with torch.inference_mode():
        for index, value in enumerate(inputs):
            nchw = torch.from_numpy(value.transpose(0, 3, 1, 2).copy())
            native_features = wrapper(nchw).numpy()
            expected = student(nchw)[0].numpy()
            checks.append(compare(expected, student.embedding_head(torch.from_numpy(native_features)).numpy(), f"original_vs_split_{index}", .999999, 1e-5))
            tf_features = deployment.encode(tf.constant(value)).numpy()
            checks.append(compare(native_features, tf_features, f"pytorch_vs_tensorflow_features_{index}", .99999, .005))
            lite.set_tensor(input_spec["index"], value)
            lite.invoke()
            features = lite.get_tensor(output_spec["index"])
            actual = student.embedding_head(torch.from_numpy(features)).numpy()
            checks.append(compare(expected, actual, f"pytorch_vs_litert_embedding_{index}", .99999, .0005))
            if index == 0:
                golden = expected
    with tempfile.TemporaryDirectory(prefix="f6-assets-") as temporary:
        staged = Path(temporary)
        (staged / "f6_backbone.tflite").write_bytes(model_bytes)
        (staged / "f6_head.bin").write_bytes(head_bytes(payload["student_model_state_dict"]))
        (staged / "golden_input.bin").write_bytes(inputs[0].astype("<f4").tobytes())
        (staged / "golden_embedding.bin").write_bytes(golden.astype("<f4").tobytes())
        hashes = {p.name: sha256(p) for p in staged.iterdir()}
        identity = {"deployment_schema": 1, "preprocessing": "f6_fit_pad_opencv_nhwc_v1", "asset_sha256": hashes}
        model_id = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        manifest = {**identity, "model_id": model_id, "export_parity_passed": True, "embedding_dim": 384, "backbone_feature_dim": 960, "input_shape": [1, 800, 384, 3], "backend": "litert_gpu_backbone_cpu_head", "weights_sha256": provenance["weights_sha256"], "checkpoint_step": payload["step"], "normalization_mean": [.485, .456, .406], "normalization_std": [.229, .224, .225], "padding_rgb": [124, 116, 104], "gpu_operator_allowlist": ops, "pytorch_version": torch.__version__, "timm_version": timm.__version__, "tensorflow_version": tf.__version__, "parity": checks, "gpu_on_device_validated": False}
        for path in staged.iterdir():
            shutil.copy2(path, args.output / path.name)
        # The success manifest is the final commit marker.
        manifest_file = args.output / "f6_manifest.json.tmp"
        manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        manifest_file.replace(args.output / "f6_manifest.json")
    print(f"Export passed {len(checks)} checks. model_id={model_id}\nAssets: {args.output}\nPhone GPU parity and device acceptance are still required.")


if __name__ == "__main__":
    main()
