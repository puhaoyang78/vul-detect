"""Ordinary global VAT with tied sliding-window source perturbations."""

from __future__ import annotations
import torch
from .model import auxiliary_mode

VAT_CONFIG = dict(epsilon=1.0, xi=0.001, power_iterations=1)
VAT_DIRECTIONS = ("finite_difference", "logit")


def expand_vat_groups(values, group_ids):
    if not len(values):
        return values.new_zeros((*group_ids.shape, values.shape[-1]))
    return values[group_ids.clamp_min(0)] * group_ids.ge(0).unsqueeze(-1)


def normalize_vat_groups(values, group_ids, group_owners, window_owners, function_count, epsilon):
    expanded = expand_vat_groups(values, group_ids)
    # VAT's finite-difference gradient may be tiny but nonzero. Squaring it
    # in float32 can underflow, misstate the radius or discard the direction.
    squared = expanded.new_zeros(function_count, dtype=torch.float64).index_add(
        0, window_owners, expanded.double().square().sum((1, 2))
    )
    norms = squared.sqrt()
    scale = torch.where(
        norms > 0, epsilon / norms.clamp_min(torch.finfo(norms.dtype).tiny), torch.zeros_like(norms)
    )
    return (values.double() * scale[group_owners, None]).to(values.dtype), norms


def bernoulli_vat_kl(reference, perturbed):
    """KL(Bernoulli(p_detached) || Bernoulli(q)); no true-label input."""
    reference = reference.detach().double()
    perturbed = perturbed.double()
    p = reference.sigmoid()
    return p * (
        torch.nn.functional.logsigmoid(reference) - torch.nn.functional.logsigmoid(perturbed)
    ) + (1 - p) * (
        torch.nn.functional.logsigmoid(-reference) - torch.nn.functional.logsigmoid(-perturbed)
    )


def vat_loss(
    model, records, builder, *, device, noise_seed,
    direction_method="finite_difference", return_delta=False,
):
    if direction_method not in VAT_DIRECTIONS:
        raise ValueError("unknown VAT direction method")
    batch, group_ids, group_owners, eligible = builder.vat_batch(records, device=device)
    ids, mask, owners, pooling, n = batch
    hidden = model.encoder.config.hidden_size
    if not len(group_owners):
        result = torch.zeros(n, device=device)
        info = dict(
            eligible=0,
            zero_direction=0,
            kl=0.0,
            norm_min=None,
            norm_max=None,
            function_forwards=0,
            encoder_sequences=0,
            encoder_token_slots=0,
            encoder_active_tokens=0,
            direction_method=direction_method,
            zero_initial_projection=0 if direction_method == "logit" else None,
        )
        return (
            (result, info, torch.zeros((*ids.shape, hidden), device=device))
            if return_delta
            else (result, info)
        )
    generator = torch.Generator(device=device).manual_seed(noise_seed)
    direction = torch.randn(
        (len(group_owners), hidden), device=device, generator=generator, dtype=torch.float32
    )
    direction, _ = normalize_vat_groups(direction, group_ids, group_owners, owners, n, 1.0)
    direction = direction.detach().requires_grad_(True)
    zero_projection = None
    with auxiliary_mode(model):
        if direction_method == "finite_difference":
            with torch.no_grad():
                reference = model(*batch, force_fp32=True)
            probe = expand_vat_groups(direction, group_ids) * VAT_CONFIG["xi"]
            near = model(*batch, embedding_delta=probe, force_fp32=True)
            divergence = bernoulli_vat_kl(reference, near)
            gradient = torch.autograd.grad(divergence.sum(), direction, only_inputs=True)[0].detach()
            forwards = 3
        else:
            center = torch.zeros_like(direction, requires_grad=True)
            center_logits = model(
                *batch, embedding_delta=expand_vat_groups(center, group_ids), force_fp32=True
            )
            reference = center_logits.detach()
            gradient = torch.autograd.grad(center_logits.sum(), center, only_inputs=True)[0].detach()
            # H d = p(1-p) g (g^T d). Choose its sign in the ORIGINAL
            # tied coordinates, before projecting through (A^T A)^-1.
            projection = gradient.new_zeros(n, dtype=torch.float64).index_add(
                0, group_owners, (gradient.double() * direction.detach().double()).sum(1)
            )
            zero_projection = sum(e and float(projection[i]) == 0 for i, e in enumerate(eligible))
            gradient = gradient * projection.sign()[group_owners, None].to(gradient.dtype)
            forwards = 2
        if not torch.isfinite(gradient).all():
            raise ValueError("nonfinite VAT direction gradient")
        # The tied embedding subspace has metric A^T A = occurrence counts.
        # Autograd sums tied gradients; divide by counts to project the physical
        # token-space gradient, then normalize in the expanded token space.
        counts = torch.bincount(group_ids[group_ids >= 0], minlength=len(group_owners)).float()
        gradient = gradient / counts[:, None]
        direction, norms = normalize_vat_groups(
            gradient, group_ids, group_owners, owners, n, VAT_CONFIG["epsilon"]
        )
        delta = expand_vat_groups(direction.detach(), group_ids)
        perturbed = model(*batch, embedding_delta=delta, force_fp32=True)
        losses = bernoulli_vat_kl(reference, perturbed).float()
    if torch.any(losses < -1e-7) or not torch.isfinite(losses).all():
        raise ValueError("invalid VAT KL")
    losses = losses.clamp_min(0) * torch.tensor(eligible, device=device)
    squared = delta.double().square().sum((1, 2))
    actual = squared.new_zeros(n).index_add(0, owners, squared).sqrt()
    active = actual[actual > 0]
    if not torch.allclose(
        active, active.new_full(active.shape, VAT_CONFIG["epsilon"]), rtol=1e-5, atol=1e-7
    ):
        raise ValueError("expanded VAT perturbations do not match the per-function radius")
    info = dict(
        eligible=sum(eligible),
        zero_direction=sum(e and float(norms[i]) == 0 for i, e in enumerate(eligible)),
        kl=float(losses.detach().sum()),
        norm_min=float(active.min()) if len(active) else None,
        norm_max=float(active.max()) if len(active) else None,
        function_forwards=forwards * n,
        encoder_sequences=forwards * len(ids),
        encoder_token_slots=forwards * ids.numel(),
        encoder_active_tokens=forwards * int(mask.sum()),
        direction_method=direction_method,
        zero_initial_projection=zero_projection,
    )
    if return_delta:
        info["reference_logits"] = reference.detach().cpu().tolist()
        if direction_method == "finite_difference":
            info["probe_logit_change"] = (near.detach() - reference).cpu().tolist()
        else:
            info["raw_initial_projection"] = projection.detach().cpu().tolist()
    return (losses, info, delta.detach()) if return_delta else (losses, info)
