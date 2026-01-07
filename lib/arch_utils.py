import torch
import torch.nn as nn

"""Shared neural-operator building blocks for PI-GANO.

This module is intentionally model-agnostic and is used by both
`lib.model_darcy` and `lib.model_plate` to reduce duplication.

It includes:
- Small MLP builders
- Masked geometry encoders
- Common base classes and gated-modulation helpers
"""


# =========================================================================
# Utility Functions
# =========================================================================

def build_mlp(
    in_dim,
    fc_dim,
    n_layers,
    out_dim=None,
    activation=nn.Tanh,
    include_output_layer=True,
):
    """Build an MLP as a list of layers suitable for `nn.Sequential`."""
    if n_layers <= 0:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    if out_dim is None:
        out_dim = fc_dim

    layers = [nn.Linear(in_dim, fc_dim), activation()]
    for _ in range(n_layers - 1):
        layers.extend([nn.Linear(fc_dim, fc_dim), activation()])
    if include_output_layer:
        layers.append(nn.Linear(fc_dim, out_dim))

    return layers


def build_sequential_layers(in_dim, hidden_dim, n_layers):
    """Return a list of `nn.Linear` layers: [FC1, FC2, ...]."""
    layers = [nn.Linear(in_dim, hidden_dim)]
    for _ in range(n_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
    return layers


# =========================================================================
# Base Architecture Classes
# =========================================================================


class BaseNeuralOperator(nn.Module):
    """Base class for all physics-informed neural operator models."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc_dim = config["model"]["fc_dim"]
        self.n_layer = config["model"]["N_layer"]


class BlendMixin:
    """Shared primitives for mix-in based forward helpers."""

    def _linear_act(self, linear, x):
        x = linear(x)
        x = self.act(x)
        return x


class DCONBlendMixin(BlendMixin):
    """Shared forward helpers for DCON/GANO-style gated modulation.

    Provides both 3-layer and 4-layer trunk helpers.
    """

    @staticmethod
    def _blend(x, enc):
        return x * enc

    def _linear_act_blend(self, linear, x, enc):
        x = self._linear_act(linear, x)
        return self._blend(x, enc)

    def _predict_head(self, inp, enc, fcs, *, final_gate=True, reduce_mean=True, gate_layers=None):
        """Apply gated-modulation trunk head.

        This supports both the 3-layer (Darcy/DCON) and 4-layer (Plate/GANO)
        variants by accepting an ordered sequence of Linear layers.

        Args:
            inp: (B, M, Din)
            enc: (B, 1, Denc)
            fcs: sequence of Linear layers, length >= 2
            final_gate: if True, apply a final multiplicative gate with `enc`
            gate_layers: controls which intermediate layers apply the multiplicative gate.
                - None: gate after every pre-final layer (default)
                - int k: gate only after the first k pre-final layers
            reduce_mean: if True, return mean over feature dim (B, M).
                If False, return raw features (B, M, Fout).

        Returns:
            (B, M) if reduce_mean else (B, M, Fout)
        """
        if len(fcs) < 2:
            raise ValueError(f"fcs must have >=2 layers, got {len(fcs)}")

        if gate_layers is None:
            gate_layers = len(fcs) - 1

        x = inp
        for idx, fc in enumerate(fcs[:-1]):
            x = self._linear_act(fc, x)
            if idx < gate_layers:
                x = self._blend(x, enc)

        x = fcs[-1](x)
        if final_gate:
            x = self._blend(x, enc)
        if reduce_mean:
            return torch.mean(x, -1)
        return x


class BaseDCON(DCONBlendMixin, BaseNeuralOperator):
    """Unified masked-DCON base for both Darcy (scalar) and Plate (vector).

    Mirrors the PI-DCON pattern of distinguishing scalar/vector outputs via
    `field_dim`, while preserving PI-GANO's masked max-pooling over boundary
    samples (via `par_flag`).

    Args:
        field_dim: 1 for scalar field, 2 for 2-component vector field
        par_dim: boundary input dimension (e.g., 3 for Darcy, 4 for Plate)
    """

    def __init__(self, field_dim, config, par_dim):
        super().__init__(config)
        if field_dim not in (1, 2):
            raise ValueError(f"field_dim must be 1 or 2, got {field_dim}")

        self.field_dim = field_dim
        self.par_dim = par_dim

        # Branch network (shared)
        self.branch = nn.Sequential(*build_mlp(par_dim, self.fc_dim, self.n_layer, self.fc_dim))

        # Trunk networks (3 layers per component, like PI-DCON)
        self.FC = nn.ModuleList(
            [nn.ModuleList(build_sequential_layers(2, self.fc_dim, 3)) for _ in range(field_dim)]
        )

        self.act = nn.Tanh()

    def _encode_par(self, par, par_flag):
        enc = self.branch(par)
        enc_masked = enc * par_flag.unsqueeze(-1)
        return torch.amax(enc_masked, 1, keepdim=True)

    def _zip(self, xy, par, par_flag, axis=0, enc=None):
        if enc is None:
            enc = self._encode_par(par, par_flag)
        return self._predict_head(xy, enc, self.FC[axis])


# =========================================================================
# Geometry Encoders
# =========================================================================


class DomainGeometry(nn.Module):
    """Domain geometry encoder using point-wise features (outputs F)."""

    def __init__(self, config):
        super().__init__()
        fc_dim = config["model"]["fc_dim"]
        n_layer = config["model"]["N_layer"]
        self.branch = nn.Sequential(*build_mlp(2, fc_dim, n_layer, fc_dim))

    def forward(self, shape_coor, shape_flag):
        """Compute masked mean embedding.

        Args:
            shape_coor: (B, M, 2)
            shape_flag: (B, M)

        Returns:
            Domain_enc: (B, 1, F)
        """
        enc = self.branch(shape_coor)  # (B, M, F)
        enc_masked = enc * shape_flag.unsqueeze(-1)
        denom = torch.sum(shape_flag.unsqueeze(-1), 1, keepdim=True)
        # Avoid NaNs if a sample has all flags == 0
        denom = torch.clamp(denom, min=1.0)
        return torch.sum(enc_masked, 1, keepdim=True) / denom


class DomainGeometryOtherEmbedding(nn.Module):
    """Geometry encoder variant producing 2*F (for add/mul coupling variants)."""

    def __init__(self, config):
        super().__init__()
        fc_dim = config["model"]["fc_dim"]
        n_layer = config["model"]["N_layer"]
        self.branch = nn.Sequential(*build_mlp(2, fc_dim, n_layer, 2 * fc_dim))

    def forward(self, shape_coor, shape_flag):
        """Compute masked mean embedding.

        Returns:
            Domain_enc: (B, 1, 2F)
        """
        enc = self.branch(shape_coor)  # (B, M, 2F)
        enc_masked = enc * shape_flag.unsqueeze(-1)
        denom = torch.sum(shape_flag.unsqueeze(-1), 1, keepdim=True)
        denom = torch.clamp(denom, min=1.0)
        return torch.sum(enc_masked, 1, keepdim=True) / denom


# =========================================================================
# Geometry-aware Operator Base
# =========================================================================


class BaseGANO(DCONBlendMixin, BaseNeuralOperator):
    """Unified base for PI-GANO variants (Darcy + Plate).

    This base centralizes:
    - geometry embedding via `DG`
    - masked max-pooling parameter encoding via `branch`
    - feature fusion helpers (`_combine_embeddings`)
    - a unified `_zip` that supports scalar/vector fields via `field_dim`

    Subclasses are expected to define:
    - `self.branch`: boundary/parameter encoder network
    - `self.FC`: trunk layers
        - if field_dim == 1: `self.FC` is a sequence of Linear layers
        - if field_dim == 2: `self.FC` is a sequence of sequences (per component)
    - any coordinate lifting layers (optional)
    """

    def __init__(self, field_dim, config, par_dim):
        super().__init__(config)
        if field_dim not in (1, 2):
            raise ValueError(f"field_dim must be 1 or 2, got {field_dim}")

        self.field_dim = field_dim
        self.par_dim = par_dim

        # Default geometry encoder (subclasses may override with DomainGeometryOtherEmbedding)
        self.DG = DomainGeometry(config)
        self.act = nn.Tanh()

    def _encode_geometry(self, shape_coor, shape_flag):
        return self.DG(shape_coor, shape_flag)

    def _encode_par(self, par, par_flag):
        enc = self.branch(par)
        enc_masked = enc * par_flag.unsqueeze(-1)
        return torch.amax(enc_masked, 1, keepdim=True)

    def _combine_embeddings(self, xy_local, domain_enc, coupling="concat"):
        """Fuse local coordinate features with a global domain embedding."""
        _, mD = xy_local.shape[:2]
        if coupling == "concat":
            return torch.cat((xy_local, domain_enc.repeat(1, mD, 1)), -1)
        if coupling == "add":
            return xy_local + domain_enc.repeat(1, mD, 1)
        if coupling == "mul":
            return xy_local * domain_enc.repeat(1, mD, 1)
        raise ValueError(f"Unknown coupling: {coupling}")

    def _zip(self, xy_global, par, par_flag, axis=0, enc=None):
        """Predict one component using the trunk + gated modulation."""
        if enc is None:
            enc = self._encode_par(par, par_flag)

        if self.field_dim == 1:
            fcs = self.FC
        else:
            fcs = self.FC[axis]

        # Legacy plate-GANO applies enc-gating after the first two layers,
        # then applies a final gate before reducing.
        gate_layers = 2 if len(fcs) == 4 else None
        return self._predict_head(xy_global, enc, fcs, gate_layers=gate_layers)
