from __future__ import annotations

import types
from typing import Any

import torch


def enable_dropout_only(model: torch.nn.Module) -> None:
    """Enable dropout at inference time without enabling BN/other training behavior.

    Typical MC-Dropout usage:
      model.eval(); enable_dropout_only(model); then run multiple forward passes.
    """

    for m in model.modules():
        if isinstance(
            m,
            (
                torch.nn.Dropout,
                torch.nn.Dropout2d,
                torch.nn.Dropout3d,
                torch.nn.AlphaDropout,
            ),
        ):
            m.train()


def _as_p(x: float) -> float:
    p = float(x)
    if p < 0.0 or p >= 1.0:
        raise ValueError(f"dropout p must be in [0,1): got {p}")
    return p


def inject_stage_and_head_dropout(
    model: torch.nn.Module,
    *,
    arch: str,
    p_stage: float,
    p_head: float,
) -> torch.nn.Module:
    """Inject dropout after the last feature stage and before the classifier head.

    This is implemented by:
      - registering new dropout modules as extra attributes (no parameters)
      - overriding forward() to apply them

    This preserves state_dict parameter keys for the original backbone.
    """

    p_stage_f = _as_p(float(p_stage))
    p_head_f = _as_p(float(p_head))
    if p_stage_f <= 0.0 and p_head_f <= 0.0:
        return model

    # Register dropout modules (no params, so checkpoint compatibility is preserved).
    model._mc_dropout_stage = (
        torch.nn.Dropout2d(p_stage_f) if p_stage_f > 0.0 else torch.nn.Identity()
    )  # type: ignore[attr-defined]
    model._mc_dropout_head = (
        torch.nn.Dropout(p_head_f) if p_head_f > 0.0 else torch.nn.Identity()
    )  # type: ignore[attr-defined]

    arch_s = str(arch).strip().lower()

    if arch_s == "resnet18":

        def _forward(self: Any, x: torch.Tensor) -> torch.Tensor:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self._mc_dropout_stage(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self._mc_dropout_head(x)
            x = self.fc(x)
            return x

        model.forward = types.MethodType(_forward, model)
        return model

    if arch_s == "efficientnet_b0":

        def _forward(self: Any, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self._mc_dropout_stage(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self._mc_dropout_head(x)
            x = self.classifier(x)
            return x

        model.forward = types.MethodType(_forward, model)
        return model

    if arch_s == "convnext_tiny":
        # ConvNeXt forward in torchvision is: features -> avgpool -> classifier (LN -> Flatten -> Linear)
        classifier = getattr(model, "classifier", None)

        def _forward(self: Any, x: torch.Tensor) -> torch.Tensor:
            x = self.features(x)
            x = self._mc_dropout_stage(x)
            x = self.avgpool(x)

            clf = classifier if classifier is not None else self.classifier
            if isinstance(clf, torch.nn.Sequential) and len(clf) >= 3:
                x = clf[0](x)
                x = clf[1](x)
                x = self._mc_dropout_head(x)
                x = clf[2](x)
                return x

            # Fallback: best-effort
            x = torch.flatten(x, 1)
            x = self._mc_dropout_head(x)
            x = self.classifier(x)
            return x

        model.forward = types.MethodType(_forward, model)
        return model

    raise ValueError(f"Unsupported arch for dropout injection: {arch}. Use resnet18 | efficientnet_b0 | convnext_tiny")
