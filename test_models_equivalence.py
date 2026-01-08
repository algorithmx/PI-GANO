import importlib.util
import os
import unittest

import torch


def _load_bak_module(module_name, bak_path):
    """Load a backup module from the given path."""
    spec = importlib.util.spec_from_file_location(module_name, bak_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _signature(model: torch.nn.Module):
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items()}


def _count_parameters(model: torch.nn.Module, ignore_prefixes=()):
    """Count total and trainable parameters in a model.

    Note: Some legacy/bak models define parameters that are never used in forward
    (e.g. FC4u/FC5u heads). For a fair comparison, callers can pass
    `ignore_prefixes` to exclude those by name.
    """

    def _keep(name: str) -> bool:
        return not any(name.startswith(pfx) for pfx in ignore_prefixes)

    total = sum(p.numel() for name, p in model.named_parameters() if _keep(name))
    trainable = sum(
        p.numel() for name, p in model.named_parameters() if _keep(name) and p.requires_grad
    )
    return {"total": total, "trainable": trainable}


def _parameter_breakdown(model: torch.nn.Module):
    """Get detailed parameter counts by layer name."""
    return {
        name: {"shape": tuple(p.shape), "count": p.numel()}
        for name, p in model.named_parameters()
    }


# ============================================================================
# State Dict Mapping Functions - Darcy Models
# ============================================================================

def map_pi_dcon_state_dict(new_state_dict):
    """Map new PI_DCON state dict keys to old format.

    New (older refactor): FC.0.*, FC.1.*, FC.2.* (ModuleList of layers)
    New (current): FC.0.0.*, FC.0.1.*, FC.0.2.* (field_dim=1 -> nested ModuleList)
    Old: FC1u.*, FC2u.*, FC3u.* (direct attributes)
    """
    old_state_dict = {}
    for key, value in new_state_dict.items():
        if key.startswith("FC.0.0."):
            # FC.0.0.* -> FC1u.*
            old_key = key.replace("FC.0.0.", "FC1u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.1."):
            # FC.0.1.* -> FC2u.*
            old_key = key.replace("FC.0.1.", "FC2u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.2."):
            # FC.0.2.* -> FC3u.*
            old_key = key.replace("FC.0.2.", "FC3u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0."):
            # FC.0.* -> FC1u.*
            old_key = key.replace("FC.0.", "FC1u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.1."):
            # FC.1.* -> FC2u.*
            old_key = key.replace("FC.1.", "FC2u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.2."):
            # FC.2.* -> FC3u.*
            old_key = key.replace("FC.2.", "FC3u.")
            old_state_dict[old_key] = value
        else:
            old_state_dict[key] = value
    return old_state_dict


def map_pi_pn_state_dict(new_state_dict):
    """Map new PI_PN state dict keys to old format.
    Keys should already match since both use Sequential.
    """
    return new_state_dict.copy()


def map_pi_gano_state_dict(new_state_dict):
    """Map new PI_GANO state dict keys to old format.

    New (older refactor): FC.0.*, FC.1.*, FC.2.*, DG.branch.*
    New (current): FC.0.0.*, FC.0.1.*, FC.0.2.* (scalar -> nested ModuleList)
    Old: FC1u.*, FC2u.*, FC3u.*, DG.branch.*
    """
    old_state_dict = {}
    for key, value in new_state_dict.items():
        if key.startswith("FC.0.0."):
            old_state_dict[key.replace("FC.0.0.", "FC1u.")] = value
        elif key.startswith("FC.0.1."):
            old_state_dict[key.replace("FC.0.1.", "FC2u.")] = value
        elif key.startswith("FC.0.2."):
            old_state_dict[key.replace("FC.0.2.", "FC3u.")] = value
        elif key.startswith("FC.0."):
            old_state_dict[key.replace("FC.0.", "FC1u.")] = value
        elif key.startswith("FC.1."):
            old_state_dict[key.replace("FC.1.", "FC2u.")] = value
        elif key.startswith("FC.2."):
            old_state_dict[key.replace("FC.2.", "FC3u.")] = value
        else:
            old_state_dict[key] = value
    return old_state_dict


def map_pi_gano_add_state_dict(new_state_dict):
    """Map new PI_GANO_add state dict keys to old format.

    Similar to PI_GANO but with 2*fc_dim dimensions.
    """
    return map_pi_gano_state_dict(new_state_dict)


def map_pi_gano_mul_state_dict(new_state_dict):
    """Map new PI_GANO_mul state dict keys to old format.
    Same as PI_GANO_add.
    """
    return map_pi_gano_add_state_dict(new_state_dict)


# ============================================================================
# State Dict Mapping Functions - Plate Models
# ============================================================================

def map_pi_dcon_plate_state_dict(new_state_dict):
    """Map new PI_DCON_plate state dict keys to old format.

    New: FC.0.0.*, FC.0.1.*, FC.0.2.*, FC.1.0.*, FC.1.1.*, FC.1.2.*
    Old: FC1u.*, FC2u.*, FC3u.*, FC1v.*, FC2v.*, FC3v.*
    """
    old_state_dict = {}
    for key, value in new_state_dict.items():
        # u-component (field_dim 0)
        if key.startswith("FC.0.0."):
            old_key = key.replace("FC.0.0.", "FC1u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.1."):
            old_key = key.replace("FC.0.1.", "FC2u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.2."):
            old_key = key.replace("FC.0.2.", "FC3u.")
            old_state_dict[old_key] = value
        # v-component (field_dim 1)
        elif key.startswith("FC.1.0."):
            old_key = key.replace("FC.1.0.", "FC1v.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.1.1."):
            old_key = key.replace("FC.1.1.", "FC2v.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.1.2."):
            old_key = key.replace("FC.1.2.", "FC3v.")
            old_state_dict[old_key] = value
        else:
            old_state_dict[key] = value
    return old_state_dict


def map_pi_pn_plate_state_dict(new_state_dict):
    """Map new PI_PN_plate state dict keys to old format.
    Keys should already match since both use Sequential.
    """
    return new_state_dict.copy()


def map_pi_gano_plate_state_dict(new_state_dict):
    """Map new PI_GANO_plate state dict keys to old format.

    New: FC.0.0.*, FC.0.1.*, FC.0.2.*, FC.0.3.*, FC.1.0.*, FC.1.1.*, FC.1.2.*, FC.1.3.*
    Old: FC1u.*, FC2u.*, FC3u.*, FC4u.*, FC1v.*, FC2v.*, FC3v.*, FC4v.*, DG.branch.*
    """
    old_state_dict = {}
    for key, value in new_state_dict.items():
        # u-component (4 layers for plate)
        if key.startswith("FC.0.0."):
            old_key = key.replace("FC.0.0.", "FC1u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.1."):
            old_key = key.replace("FC.0.1.", "FC2u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.2."):
            old_key = key.replace("FC.0.2.", "FC3u.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.0.3."):
            old_key = key.replace("FC.0.3.", "FC4u.")
            old_state_dict[old_key] = value
        # v-component
        elif key.startswith("FC.1.0."):
            old_key = key.replace("FC.1.0.", "FC1v.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.1.1."):
            old_key = key.replace("FC.1.1.", "FC2v.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.1.2."):
            old_key = key.replace("FC.1.2.", "FC3v.")
            old_state_dict[old_key] = value
        elif key.startswith("FC.1.3."):
            old_key = key.replace("FC.1.3.", "FC4v.")
            old_state_dict[old_key] = value
        else:
            old_state_dict[key] = value
    return old_state_dict


def map_pi_gano_add_plate_state_dict(new_state_dict):
    """Map new PI_GANO_add_plate state dict keys to old format.
    Same as PI_GANO_plate.
    """
    return map_pi_gano_plate_state_dict(new_state_dict)


def map_pi_gano_mul_plate_state_dict(new_state_dict):
    """Map new PI_GANO_mul_plate state dict keys to old format.
    Same as PI_GANO_plate.
    """
    return map_pi_gano_plate_state_dict(new_state_dict)


def translate_state_dict_for_model(state_dict, model_type):
    """Translate new model state dict to old model format.

    Args:
        state_dict: State dict from new model
        model_type: Type of model (e.g., 'pi_dcon', 'pi_gano', 'pi_gano_plate', etc.)

    Returns:
        State dict with keys translated to old format
    """
    mapping = {
        'pi_dcon': map_pi_dcon_state_dict,
        'pi_pn': map_pi_pn_state_dict,
        'pi_pn_only_geo': map_pi_pn_state_dict,
        'pi_gano': map_pi_gano_state_dict,
        'pi_gano_add': map_pi_gano_add_state_dict,
        'pi_gano_mul': map_pi_gano_mul_state_dict,
        'pi_dcon_plate': map_pi_dcon_plate_state_dict,
        'pi_pn_plate': map_pi_pn_plate_state_dict,
        'pi_pn_only_geo_plate': map_pi_pn_plate_state_dict,
        'pi_gano_plate': map_pi_gano_plate_state_dict,
        'pi_gano_add_plate': map_pi_gano_add_plate_state_dict,
        'pi_gano_mul_plate': map_pi_gano_mul_plate_state_dict,
    }

    mapper = mapping.get(model_type, lambda x: x.copy())
    return mapper(state_dict)


def _signature_from_dict(state_dict):
    """Create signature dict from state dict (for comparing translated state dicts)."""
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in state_dict.items()}


# ============================================================================
# Test Classes
# ============================================================================

class TestDarcyModelsForwardEquivalence(unittest.TestCase):
    """Test forward pass equivalence for Darcy flow models."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))

        # Load backup modules
        bak_darcy_path = os.path.join(cls.this_dir, "lib", "model_darcy.bak.py")
        cls.models_darcy_bak = _load_bak_module("model_darcy_bak", bak_darcy_path)

        # Import new models
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_darcy as models_darcy_new  # noqa: E402
        cls.models_darcy_new = models_darcy_new

    def _make_inputs(self, B=3, M=17, N=13, M_shape=11, device="cpu", dtype=torch.float32):
        """Create test inputs for Darcy models."""
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        par_flag = torch.randint(0, 2, (B, N), generator=g, device=device, dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device=device, dtype=dtype)
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device=device, dtype=torch.float32)

        # Avoid all-zero flags which cause undefined behavior (division by zero) in bak DG.
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0
        return x_coor, y_coor, par, par_flag, shape_coor, shape_flag

    def _make_inputs_pn(self, B=3, M=17, device="cpu", dtype=torch.float32):
        """Create PointNet-style inputs with consistent shapes (B, M)."""
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par_val = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        flag = torch.randint(0, 2, (B, M), generator=g, device=device, dtype=torch.float32)
        flag[:, 0] = 1.0
        return x_coor, y_coor, par_val, flag

    def _assert_close(self, a, b, msg=""):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7, msg=msg)

    def _compare_forward_single_output(
        self, new_model, bak_model, inputs, model_type
    ):
        """Compare forward pass for models with single output."""
        x_coor, y_coor, par, par_flag, shape_coor, shape_flag = inputs

        # Translate new model state dict to old format
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)

        # Load translated state dict into old model
        bak_model.load_state_dict(translated_state, strict=False)

        new_model.eval()
        bak_model.eval()

        with torch.no_grad():
            out_new = new_model(x_coor, y_coor, par, par_flag, shape_coor, shape_flag)
            out_bak = bak_model(x_coor, y_coor, par, par_flag, shape_coor, shape_flag)

        self._assert_close(out_new, out_bak, msg=f"Forward pass mismatch for {model_type}")

    def _get_model_type(self, model):
        """Get model type string for state dict translation."""
        class_name = model.__class__.__name__
        mapping = {
            'PI_DCON': 'pi_dcon',
            'PI_PN': 'pi_pn',
            'PI_PN_only_geo': 'pi_pn_only_geo',
            'PI_GANO': 'pi_gano',
            'PI_GANO_add': 'pi_gano_add',
            'PI_GANO_mul': 'pi_gano_mul',
        }
        return mapping.get(class_name, None)

    def test_pi_dcon_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()

        new = self.models_darcy_new.PI_DCON(config)
        bak = self.models_darcy_bak.PI_DCON(config)
        self._compare_forward_single_output(new, bak, inputs, 'pi_dcon')

    def test_pi_pn_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par_val, flag = self._make_inputs_pn()

        new = self.models_darcy_new.PI_PN(config)
        bak = self.models_darcy_bak.PI_PN(config)

        # PI_PN has different signature (no shape_coor, shape_flag)
        new_state = new.state_dict()
        translated_state = translate_state_dict_for_model(new_state, 'pi_pn')
        bak.load_state_dict(translated_state, strict=False)

        new.eval()
        bak.eval()

        with torch.no_grad():
            out_new = new(x_coor, y_coor, par_val, flag)
            out_bak = bak(x_coor, y_coor, par_val, flag)

        self._assert_close(out_new, out_bak)

    def test_pi_pn_only_geo_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, par_val, flag = self._make_inputs_pn()

        new = self.models_darcy_new.PI_PN_only_geo(config)
        bak = self.models_darcy_bak.PI_PN_only_geo(config)

        new_state = new.state_dict()
        translated_state = translate_state_dict_for_model(new_state, 'pi_pn_only_geo')
        bak.load_state_dict(translated_state, strict=False)

        new.eval()
        bak.eval()

        with torch.no_grad():
            out_new = new(x_coor, y_coor, par_val, flag)
            out_bak = bak(x_coor, y_coor, par_val, flag)

        self._assert_close(out_new, out_bak)

    def test_pi_gano_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()

        new = self.models_darcy_new.PI_GANO(config)
        bak = self.models_darcy_bak.PI_GANO(config)
        self._compare_forward_single_output(new, bak, inputs, 'pi_gano')

    def test_pi_gano_add_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()

        new = self.models_darcy_new.PI_GANO_add(config)
        bak = self.models_darcy_bak.PI_GANO_add(config)
        self._compare_forward_single_output(new, bak, inputs, 'pi_gano_add')

    def test_pi_gano_mul_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()

        new = self.models_darcy_new.PI_GANO_mul(config)
        bak = self.models_darcy_bak.PI_GANO_mul(config)
        self._compare_forward_single_output(new, bak, inputs, 'pi_gano_mul')


class TestPlateModelsForwardEquivalence(unittest.TestCase):
    """Test forward pass equivalence for plate stress models."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))

        # Load backup modules
        bak_plate_path = os.path.join(cls.this_dir, "lib", "model_plate.bak.py")
        cls.models_plate_bak = _load_bak_module("model_plate_bak", bak_plate_path)

        # Import new models
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_plate as models_plate_new  # noqa: E402
        cls.models_plate_new = models_plate_new

    def _make_inputs(self, B=3, M=17, N=13, M_shape=11, device="cpu", dtype=torch.float32):
        """Create test inputs for Plate models (par has 4 channels)."""
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 4, generator=g, device=device, dtype=dtype)  # 4 channels for plate
        par_flag = torch.randint(0, 2, (B, N), generator=g, device=device, dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device=device, dtype=dtype)
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device=device, dtype=torch.float32)

        # Avoid all-zero flags which cause undefined behavior (division by zero) in bak DG.
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0
        return x_coor, y_coor, par, par_flag, shape_coor, shape_flag

    def _make_inputs_pipn(self, B=3, M=17, device="cpu", dtype=torch.float32):
        """Create PIPN-style inputs with consistent shapes (B, M)."""
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        u_input = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        v_input = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        flag = torch.randint(0, 2, (B, M), generator=g, device=device, dtype=torch.float32)
        flag[:, 0] = 1.0
        return x_coor, y_coor, u_input, v_input, flag

    def _assert_close(self, a, b, msg=""):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-7, msg=msg)

    def _compare_forward_two_outputs(
        self, new_model, bak_model, inputs, model_type
    ):
        """Compare forward pass for models with two outputs (u, v)."""
        x_coor, y_coor, par, par_flag, shape_coor, shape_flag = inputs

        # Translate new model state dict to old format
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)

        # Load translated state dict into old model
        bak_model.load_state_dict(translated_state, strict=False)

        new_model.eval()
        bak_model.eval()

        with torch.no_grad():
            out_new = new_model(x_coor, y_coor, par, par_flag, shape_coor, shape_flag)
            out_bak = bak_model(x_coor, y_coor, par, par_flag, shape_coor, shape_flag)

        self.assertIsInstance(out_new, tuple)
        self.assertIsInstance(out_bak, tuple)
        self.assertEqual(len(out_new), 2)
        self.assertEqual(len(out_bak), 2)

        self._assert_close(out_new[0], out_bak[0], msg=f"{model_type}: u component mismatch")
        self._assert_close(out_new[1], out_bak[1], msg=f"{model_type}: v component mismatch")

    def _compare_forward_two_outputs_pipn(
        self, new_model, bak_model, x_coor, y_coor, u_input, v_input, flag, model_type
    ):
        """Compare forward pass for PIPN models with special signature."""
        # Translate state dict
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        new_model.eval()
        bak_model.eval()

        with torch.no_grad():
            out_new = new_model(x_coor, y_coor, u_input, v_input, flag)
            out_bak = bak_model(x_coor, y_coor, u_input, v_input, flag)

        self.assertIsInstance(out_new, tuple)
        self.assertIsInstance(out_bak, tuple)
        self.assertEqual(len(out_new), 2)
        self.assertEqual(len(out_bak), 2)

        self._assert_close(out_new[0], out_bak[0])
        self._assert_close(out_new[1], out_bak[1])

    def test_pi_dcon_plate_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()

        new = self.models_plate_new.PI_DCON_plate(config)
        bak = self.models_plate_bak.DCON(config)
        self._compare_forward_two_outputs(new, bak, inputs, 'pi_dcon_plate')

    def test_pi_pn_plate_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, u_input, v_input, flag = self._make_inputs_pipn()

        new = self.models_plate_new.PI_PN_plate(config)
        bak = self.models_plate_bak.PIPN(config)
        self._compare_forward_two_outputs_pipn(
            new, bak, x_coor, y_coor, u_input, v_input, flag, 'pi_pn_plate'
        )

    def test_pi_pn_only_geo_plate_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        x_coor, y_coor, _, _, flag = self._make_inputs_pipn()

        new = self.models_plate_new.PI_PN_only_geo_plate(config)
        bak = self.models_plate_bak.PIPN_only_geo(config)
        # PIPN_only_geo takes (x_coor, y_coor, flag)
        new_state = new.state_dict()
        translated_state = translate_state_dict_for_model(new_state, 'pi_pn_only_geo_plate')
        bak.load_state_dict(translated_state, strict=False)

        new.eval()
        bak.eval()
        with torch.no_grad():
            out_new = new(x_coor, y_coor, flag)
            out_bak = bak(x_coor, y_coor, flag)

        self.assertIsInstance(out_new, tuple)
        self.assertIsInstance(out_bak, tuple)
        self.assertEqual(len(out_new), 2)
        self.assertEqual(len(out_bak), 2)
        self._assert_close(out_new[0], out_bak[0])
        self._assert_close(out_new[1], out_bak[1])

    def test_pi_gano_plate_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()

        new = self.models_plate_new.PI_GANO_plate(config)
        bak = self.models_plate_bak.GANO(config)
        self._compare_forward_two_outputs(new, bak, inputs, 'pi_gano_plate')

    def test_pi_gano_add_plate_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()

        new = self.models_plate_new.PI_GANO_add_plate(config)
        bak = self.models_plate_bak.GANO_add(config)
        self._compare_forward_two_outputs(new, bak, inputs, 'pi_gano_add_plate')

    def test_pi_gano_mul_plate_forward(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()

        new = self.models_plate_new.PI_GANO_mul_plate(config)
        bak = self.models_plate_bak.GANO_mul(config)
        self._compare_forward_two_outputs(new, bak, inputs, 'pi_gano_mul_plate')


class TestDarcyModelsParameterEquivalence(unittest.TestCase):
    """Test parameter count equivalence for Darcy models."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))

        bak_darcy_path = os.path.join(cls.this_dir, "lib", "model_darcy.bak.py")
        cls.models_darcy_bak = _load_bak_module("model_darcy_bak", bak_darcy_path)

        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_darcy as models_darcy_new  # noqa: E402
        cls.models_darcy_new = models_darcy_new

    def _compare_parameter_counts(self, new_model, bak_model, model_name, model_type):
        """Compare parameter counts between new and old models."""
        ignore = ()
        # Bak models contain unused heads in several classes; ignore for fair comparison.
        if model_type == 'pi_dcon':
            ignore = ('FC4u',)
        elif model_type in ('pi_gano', 'pi_gano_add', 'pi_gano_mul'):
            ignore = ('FC4u',)

        new_params = _count_parameters(new_model)
        bak_params = _count_parameters(bak_model, ignore_prefixes=ignore)

        self.assertEqual(
            new_params["total"],
            bak_params["total"],
            f"{model_name}: Total parameters mismatch - "
            f"new={new_params['total']:,}, old={bak_params['total']:,}",
        )

        self.assertEqual(
            new_params["trainable"],
            new_params["total"],
            f"{model_name}: Not all parameters are trainable",
        )

    def test_pi_dcon_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_darcy_new.PI_DCON(config)
        bak = self.models_darcy_bak.PI_DCON(config)
        self._compare_parameter_counts(new, bak, "PI_DCON", 'pi_dcon')

    def test_pi_pn_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_darcy_new.PI_PN(config)
        bak = self.models_darcy_bak.PI_PN(config)
        self._compare_parameter_counts(new, bak, "PI_PN", 'pi_pn')

    def test_pi_pn_only_geo_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_darcy_new.PI_PN_only_geo(config)
        bak = self.models_darcy_bak.PI_PN_only_geo(config)
        self._compare_parameter_counts(new, bak, "PI_PN_only_geo", 'pi_pn_only_geo')

    def test_pi_gano_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_darcy_new.PI_GANO(config)
        bak = self.models_darcy_bak.PI_GANO(config)
        self._compare_parameter_counts(new, bak, "PI_GANO", 'pi_gano')

    def test_pi_gano_add_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_darcy_new.PI_GANO_add(config)
        bak = self.models_darcy_bak.PI_GANO_add(config)
        self._compare_parameter_counts(new, bak, "PI_GANO_add", 'pi_gano_add')

    def test_pi_gano_mul_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_darcy_new.PI_GANO_mul(config)
        bak = self.models_darcy_bak.PI_GANO_mul(config)
        self._compare_parameter_counts(new, bak, "PI_GANO_mul", 'pi_gano_mul')


class TestPlateModelsParameterEquivalence(unittest.TestCase):
    """Test parameter count equivalence for Plate models."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))

        bak_plate_path = os.path.join(cls.this_dir, "lib", "model_plate.bak.py")
        cls.models_plate_bak = _load_bak_module("model_plate_bak", bak_plate_path)

        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_plate as models_plate_new  # noqa: E402
        cls.models_plate_new = models_plate_new

    def _compare_parameter_counts(self, new_model, bak_model, model_name, model_type):
        """Compare parameter counts between new and old models."""
        ignore = ()
        # Bak plate models contain unused heads.
        if model_type == 'pi_dcon_plate':
            ignore = ('FC4u', 'FC4v')
        elif model_type in ('pi_gano_plate', 'pi_gano_add_plate', 'pi_gano_mul_plate'):
            ignore = ('FC5u', 'FC5v')

        new_params = _count_parameters(new_model)
        bak_params = _count_parameters(bak_model, ignore_prefixes=ignore)

        self.assertEqual(
            new_params["total"],
            bak_params["total"],
            f"{model_name}: Total parameters mismatch - "
            f"new={new_params['total']:,}, old={bak_params['total']:,}",
        )

        self.assertEqual(
            new_params["trainable"],
            new_params["total"],
            f"{model_name}: Not all parameters are trainable",
        )

    def test_pi_dcon_plate_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_plate_new.PI_DCON_plate(config)
        bak = self.models_plate_bak.DCON(config)
        self._compare_parameter_counts(new, bak, "PI_DCON_plate", 'pi_dcon_plate')

    def test_pi_pn_plate_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_plate_new.PI_PN_plate(config)
        bak = self.models_plate_bak.PIPN(config)
        self._compare_parameter_counts(new, bak, "PI_PN_plate", 'pi_pn_plate')

    def test_pi_pn_only_geo_plate_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_plate_new.PI_PN_only_geo_plate(config)
        bak = self.models_plate_bak.PIPN_only_geo(config)
        self._compare_parameter_counts(new, bak, "PI_PN_only_geo_plate", 'pi_pn_only_geo_plate')

    def test_pi_gano_plate_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_plate_new.PI_GANO_plate(config)
        bak = self.models_plate_bak.GANO(config)
        self._compare_parameter_counts(new, bak, "PI_GANO_plate", 'pi_gano_plate')

    def test_pi_gano_add_plate_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_plate_new.PI_GANO_add_plate(config)
        bak = self.models_plate_bak.GANO_add(config)
        self._compare_parameter_counts(new, bak, "PI_GANO_add_plate", 'pi_gano_add_plate')

    def test_pi_gano_mul_plate_parameters(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        new = self.models_plate_new.PI_GANO_mul_plate(config)
        bak = self.models_plate_bak.GANO_mul(config)
        self._compare_parameter_counts(new, bak, "PI_GANO_mul_plate", 'pi_gano_mul_plate')


# ============================================================================
# Enhanced Equivalence Tests
# ============================================================================

class TestDarcyModelsGradientEquivalence(unittest.TestCase):
    """Test gradient/backpropagation equivalence for Darcy models."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        bak_darcy_path = os.path.join(cls.this_dir, "lib", "model_darcy.bak.py")
        cls.models_darcy_bak = _load_bak_module("model_darcy_bak", bak_darcy_path)
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_darcy as models_darcy_new
        cls.models_darcy_new = models_darcy_new

    def _make_inputs(self, B=3, M=17, N=13, M_shape=11, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype, requires_grad=True)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype, requires_grad=True)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype, requires_grad=True)
        par_flag = torch.randint(0, 2, (B, N), generator=g, device=device, dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device=device, dtype=dtype, requires_grad=True)
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device=device, dtype=torch.float32)
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0
        return x_coor, y_coor, par, par_flag, shape_coor, shape_flag

    def _compare_gradients(self, new_model, bak_model, inputs_args, model_type):
        """Compare gradients between new and old models.

        CRITICAL: Uses separate input tensors for each model to ensure
        gradient comparison is meaningful (not comparing tensor to itself).
        """
        # Create SEPARATE input tensors for each model with same seed for reproducibility
        B, M, N, M_shape = 3, 17, 13, 11
        device, dtype = "cpu", torch.float32

        # Generate inputs for NEW model
        g_new = torch.Generator(device=device)
        g_new.manual_seed(1234)
        x_coor_new = torch.randn(B, M, generator=g_new, device=device, dtype=dtype, requires_grad=True)
        y_coor_new = torch.randn(B, M, generator=g_new, device=device, dtype=dtype, requires_grad=True)
        par_new = torch.randn(B, N, 3, generator=g_new, device=device, dtype=dtype, requires_grad=True)
        par_flag_new = torch.randint(0, 2, (B, N), generator=g_new, device=device, dtype=torch.float32)
        shape_coor_new = torch.randn(B, M_shape, 2, generator=g_new, device=device, dtype=dtype, requires_grad=True)
        shape_flag_new = torch.randint(0, 2, (B, M_shape), generator=g_new, device=device, dtype=torch.float32)
        par_flag_new[:, 0] = 1.0
        shape_flag_new[:, 0] = 1.0

        # Generate inputs for BAK model (same seed for identical values)
        g_bak = torch.Generator(device=device)
        g_bak.manual_seed(1234)
        x_coor_bak = torch.randn(B, M, generator=g_bak, device=device, dtype=dtype, requires_grad=True)
        y_coor_bak = torch.randn(B, M, generator=g_bak, device=device, dtype=dtype, requires_grad=True)
        par_bak = torch.randn(B, N, 3, generator=g_bak, device=device, dtype=dtype, requires_grad=True)
        par_flag_bak = torch.randint(0, 2, (B, N), generator=g_bak, device=device, dtype=torch.float32)
        shape_coor_bak = torch.randn(B, M_shape, 2, generator=g_bak, device=device, dtype=dtype, requires_grad=True)
        shape_flag_bak = torch.randint(0, 2, (B, M_shape), generator=g_bak, device=device, dtype=torch.float32)
        par_flag_bak[:, 0] = 1.0
        shape_flag_bak[:, 0] = 1.0

        # Translate state dict and load into bak model
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        missing, unexpected = bak_model.load_state_dict(translated_state, strict=False)

        # Report any state dict loading issues (Issue #2 fix)
        if missing:
            print(f"WARNING {model_type}: Missing keys in bak model: {missing[:5]}... (showing first 5)")
        if unexpected:
            print(f"WARNING {model_type}: Unexpected keys in state dict: {unexpected[:5]}... (showing first 5)")

        new_model.train()
        bak_model.train()

        # Forward and backward for NEW model
        out_new = new_model(x_coor_new, y_coor_new, par_new, par_flag_new, shape_coor_new, shape_flag_new)
        loss_new = out_new.sum()
        loss_new.backward()

        # Forward and backward for BAK model
        out_bak = bak_model(x_coor_bak, y_coor_bak, par_bak, par_flag_bak, shape_coor_bak, shape_flag_bak)
        loss_bak = out_bak.sum()
        loss_bak.backward()

        # Compare DIFFERENT gradient tensors (the actual fix for Issue #1)
        torch.testing.assert_close(x_coor_new.grad, x_coor_bak.grad, rtol=1e-5, atol=1e-8,
                                   msg=f"{model_type}: x_coor gradient mismatch")
        torch.testing.assert_close(y_coor_new.grad, y_coor_bak.grad, rtol=1e-5, atol=1e-8,
                                   msg=f"{model_type}: y_coor gradient mismatch")
        torch.testing.assert_close(par_new.grad, par_bak.grad, rtol=1e-5, atol=1e-8,
                                   msg=f"{model_type}: par gradient mismatch")
        torch.testing.assert_close(shape_coor_new.grad, shape_coor_bak.grad, rtol=1e-5, atol=1e-8,
                                   msg=f"{model_type}: shape_coor gradient mismatch")

        # Compare parameter gradients
        new_grads = {}
        for name, param in new_model.named_parameters():
            if param.grad is not None:
                new_grads[name] = param.grad

        for name, param in bak_model.named_parameters():
            if param.grad is not None and name in new_grads:
                torch.testing.assert_close(new_grads[name], param.grad, rtol=1e-5, atol=1e-8,
                                           msg=f"{model_type}: gradient mismatch for {name}")

    def test_pi_dcon_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()
        new = self.models_darcy_new.PI_DCON(config)
        bak = self.models_darcy_bak.PI_DCON(config)
        self._compare_gradients(new, bak, inputs, 'pi_dcon')

    def test_pi_gano_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()
        new = self.models_darcy_new.PI_GANO(config)
        bak = self.models_darcy_bak.PI_GANO(config)
        self._compare_gradients(new, bak, inputs, 'pi_gano')


class TestDarcyModelsMultipleConfigurations(unittest.TestCase):
    """Test equivalence across various input configurations."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        bak_darcy_path = os.path.join(cls.this_dir, "lib", "model_darcy.bak.py")
        cls.models_darcy_bak = _load_bak_module("model_darcy_bak", bak_darcy_path)
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_darcy as models_darcy_new
        cls.models_darcy_new = models_darcy_new

    def _make_inputs(self, B=3, M=17, N=13, M_shape=11, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        par_flag = torch.randint(0, 2, (B, N), generator=g, device=device, dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device=device, dtype=dtype)
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device=device, dtype=torch.float32)
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0
        return x_coor, y_coor, par, par_flag, shape_coor, shape_flag

    def _test_config(self, new_model, bak_model, model_type, inputs):
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        new_model.eval()
        bak_model.eval()

        with torch.no_grad():
            out_new = new_model(*inputs)
            out_bak = bak_model(*inputs)

        torch.testing.assert_close(out_new, out_bak, rtol=1e-5, atol=1e-7)

    def test_pi_dcon_varying_batch_sizes(self):
        """Test with different batch sizes."""
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        for B in [1, 2, 5, 10]:
            with self.subTest(batch_size=B):
                inputs = self._make_inputs(B=B)
                new = self.models_darcy_new.PI_DCON(config)
                bak = self.models_darcy_bak.PI_DCON(config)
                self._test_config(new, bak, 'pi_dcon', inputs)

    def test_pi_dcon_varying_dimensions(self):
        """Test with different M, N, M_shape values."""
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        test_configs = [
            (3, 10, 5, 7),
            (3, 50, 20, 15),
            (3, 100, 30, 25),
        ]
        for B, M, N, M_shape in test_configs:
            with self.subTest(M=M, N=N, M_shape=M_shape):
                inputs = self._make_inputs(B=B, M=M, N=N, M_shape=M_shape)
                new = self.models_darcy_new.PI_DCON(config)
                bak = self.models_darcy_bak.PI_DCON(config)
                self._test_config(new, bak, 'pi_dcon', inputs)

    def test_pi_gano_varying_batch_sizes(self):
        """Test GANO with different batch sizes."""
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        for B in [1, 2, 5, 10]:
            with self.subTest(batch_size=B):
                inputs = self._make_inputs(B=B)
                new = self.models_darcy_new.PI_GANO(config)
                bak = self.models_darcy_bak.PI_GANO(config)
                self._test_config(new, bak, 'pi_gano', inputs)


class TestDarcyModelsDtypeDevice(unittest.TestCase):
    """Test equivalence across different dtypes and devices."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        bak_darcy_path = os.path.join(cls.this_dir, "lib", "model_darcy.bak.py")
        cls.models_darcy_bak = _load_bak_module("model_darcy_bak", bak_darcy_path)
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_darcy as models_darcy_new
        cls.models_darcy_new = models_darcy_new
        cls.cuda_available = torch.cuda.is_available()

    def _make_inputs(self, B=3, M=17, N=13, M_shape=11, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        par_flag = torch.randint(0, 2, (B, N), generator=g, device=device, dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device=device, dtype=dtype)
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device=device, dtype=torch.float32)
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0
        return x_coor, y_coor, par, par_flag, shape_coor, shape_flag

    def _test_config(self, new_model, bak_model, model_type, inputs):
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        new_model.eval()
        bak_model.eval()

        with torch.no_grad():
            out_new = new_model(*inputs)
            out_bak = bak_model(*inputs)

        torch.testing.assert_close(out_new, out_bak, rtol=1e-5, atol=1e-7)

    def test_pi_dcon_float64(self):
        """Test with float64 precision."""
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs(dtype=torch.float64)
        new = self.models_darcy_new.PI_DCON(config).to(dtype=torch.float64)
        bak = self.models_darcy_bak.PI_DCON(config).to(dtype=torch.float64)
        self._test_config(new, bak, 'pi_dcon', inputs)

    def test_pi_dcon_cuda(self):
        """Test with CUDA if available."""
        if not self.cuda_available:
            self.skipTest("CUDA not available")
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs(device="cuda")
        new = self.models_darcy_new.PI_DCON(config).cuda()
        bak = self.models_darcy_bak.PI_DCON(config).cuda()
        self._test_config(new, bak, 'pi_dcon', inputs)

    def test_pi_gano_float64(self):
        """Test GANO with float64 precision."""
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs(dtype=torch.float64)
        new = self.models_darcy_new.PI_GANO(config).to(dtype=torch.float64)
        bak = self.models_darcy_bak.PI_GANO(config).to(dtype=torch.float64)
        self._test_config(new, bak, 'pi_gano', inputs)


class TestDarcyModelsNumericalStability(unittest.TestCase):
    """Test numerical stability with extreme/near-zero values."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        bak_darcy_path = os.path.join(cls.this_dir, "lib", "model_darcy.bak.py")
        cls.models_darcy_bak = _load_bak_module("model_darcy_bak", bak_darcy_path)
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_darcy as models_darcy_new
        cls.models_darcy_new = models_darcy_new

    def _test_config(self, new_model, bak_model, model_type, inputs):
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        new_model.eval()
        bak_model.eval()

        with torch.no_grad():
            out_new = new_model(*inputs)
            out_bak = bak_model(*inputs)

        torch.testing.assert_close(out_new, out_bak, rtol=1e-4, atol=1e-6)

    def test_pi_dcon_near_zero_inputs(self):
        """Test with near-zero input values."""
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        B, M, N, M_shape = 3, 17, 13, 11
        g = torch.Generator(device="cpu")
        g.manual_seed(1234)

        x_coor = torch.randn(B, M, generator=g, device="cpu", dtype=torch.float32) * 1e-6
        y_coor = torch.randn(B, M, generator=g, device="cpu", dtype=torch.float32) * 1e-6
        par = torch.randn(B, N, 3, generator=g, device="cpu", dtype=torch.float32) * 1e-6
        par_flag = torch.randint(0, 2, (B, N), generator=g, device="cpu", dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device="cpu", dtype=torch.float32) * 1e-6
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device="cpu", dtype=torch.float32)
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0

        inputs = (x_coor, y_coor, par, par_flag, shape_coor, shape_flag)
        new = self.models_darcy_new.PI_DCON(config)
        bak = self.models_darcy_bak.PI_DCON(config)
        self._test_config(new, bak, 'pi_dcon', inputs)

    def test_pi_dcon_large_inputs(self):
        """Test with large input values."""
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        B, M, N, M_shape = 3, 17, 13, 11
        g = torch.Generator(device="cpu")
        g.manual_seed(1234)

        x_coor = torch.randn(B, M, generator=g, device="cpu", dtype=torch.float32) * 100
        y_coor = torch.randn(B, M, generator=g, device="cpu", dtype=torch.float32) * 100
        par = torch.randn(B, N, 3, generator=g, device="cpu", dtype=torch.float32) * 100
        par_flag = torch.randint(0, 2, (B, N), generator=g, device="cpu", dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device="cpu", dtype=torch.float32) * 100
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device="cpu", dtype=torch.float32)
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0

        inputs = (x_coor, y_coor, par, par_flag, shape_coor, shape_flag)
        new = self.models_darcy_new.PI_DCON(config)
        bak = self.models_darcy_bak.PI_DCON(config)
        self._test_config(new, bak, 'pi_dcon', inputs)

    def test_pi_gano_mixed_scale_inputs(self):
        """Test with mixed scale inputs (some small, some large)."""
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        B, M, N, M_shape = 3, 17, 13, 11
        g = torch.Generator(device="cpu")
        g.manual_seed(1234)

        x_coor = torch.randn(B, M, generator=g, device="cpu", dtype=torch.float32)
        y_coor = torch.randn(B, M, generator=g, device="cpu", dtype=torch.float32) * 0.01
        par = torch.randn(B, N, 3, generator=g, device="cpu", dtype=torch.float32) * 50
        par_flag = torch.randint(0, 2, (B, N), generator=g, device="cpu", dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device="cpu", dtype=torch.float32) * 1e-3
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device="cpu", dtype=torch.float32)
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0

        inputs = (x_coor, y_coor, par, par_flag, shape_coor, shape_flag)
        new = self.models_darcy_new.PI_GANO(config)
        bak = self.models_darcy_bak.PI_GANO(config)
        self._test_config(new, bak, 'pi_gano', inputs)


class TestDarcyModelsTrainingModeEquivalence(unittest.TestCase):
    """Test equivalence in training mode (affects dropout/batchnorm behavior)."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        bak_darcy_path = os.path.join(cls.this_dir, "lib", "model_darcy.bak.py")
        cls.models_darcy_bak = _load_bak_module("model_darcy_bak", bak_darcy_path)
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_darcy as models_darcy_new
        cls.models_darcy_new = models_darcy_new

    def _make_inputs(self, B=3, M=17, N=13, M_shape=11, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        par_flag = torch.randint(0, 2, (B, N), generator=g, device=device, dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device=device, dtype=dtype)
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device=device, dtype=torch.float32)
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0
        return x_coor, y_coor, par, par_flag, shape_coor, shape_flag

    def _test_training_mode(self, new_model, bak_model, model_type, inputs):
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        # Test in training mode
        new_model.train()
        bak_model.train()

        # Set same seed for deterministic behavior
        torch.manual_seed(42)
        out_new_train = new_model(*inputs)

        torch.manual_seed(42)
        out_bak_train = bak_model(*inputs)

        # Outputs should match in training mode with same seed
        torch.testing.assert_close(out_new_train, out_bak_train, rtol=1e-4, atol=1e-6,
                                   msg=f"{model_type}: training mode mismatch")

        # Test in eval mode
        new_model.eval()
        bak_model.eval()

        with torch.no_grad():
            out_new_eval = new_model(*inputs)
            out_bak_eval = bak_model(*inputs)

        torch.testing.assert_close(out_new_eval, out_bak_eval, rtol=1e-5, atol=1e-7,
                                   msg=f"{model_type}: eval mode mismatch")

    def test_pi_dcon_training_mode(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()
        new = self.models_darcy_new.PI_DCON(config)
        bak = self.models_darcy_bak.PI_DCON(config)
        self._test_training_mode(new, bak, 'pi_dcon', inputs)

    def test_pi_gano_training_mode(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()
        new = self.models_darcy_new.PI_GANO(config)
        bak = self.models_darcy_bak.PI_GANO(config)
        self._test_training_mode(new, bak, 'pi_gano', inputs)


class TestDarcyModelsBidirectionalStateDict(unittest.TestCase):
    """Test bidirectional state dict loading (new->old and old->new)."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        bak_darcy_path = os.path.join(cls.this_dir, "lib", "model_darcy.bak.py")
        cls.models_darcy_bak = _load_bak_module("model_darcy_bak", bak_darcy_path)
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_darcy as models_darcy_new
        cls.models_darcy_new = models_darcy_new

    def _make_inputs(self, B=3, M=17, N=13, M_shape=11, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 3, generator=g, device=device, dtype=dtype)
        par_flag = torch.randint(0, 2, (B, N), generator=g, device=device, dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device=device, dtype=dtype)
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device=device, dtype=torch.float32)
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0
        return x_coor, y_coor, par, par_flag, shape_coor, shape_flag

    def _invert_state_dict_mapping(self, state_dict, model_type):
        """Invert the state dict mapping (old format -> new format)."""
        inverted = {}
        if model_type == 'pi_dcon':
            for key, value in state_dict.items():
                if key.startswith("FC1u."):
                    inverted[key.replace("FC1u.", "FC.0.0.")] = value
                elif key.startswith("FC2u."):
                    inverted[key.replace("FC2u.", "FC.0.1.")] = value
                elif key.startswith("FC3u."):
                    inverted[key.replace("FC3u.", "FC.0.2.")] = value
                else:
                    inverted[key] = value
        elif model_type == 'pi_gano':
            for key, value in state_dict.items():
                if key.startswith("FC1u."):
                    inverted[key.replace("FC1u.", "FC.0.0.")] = value
                elif key.startswith("FC2u."):
                    inverted[key.replace("FC2u.", "FC.0.1.")] = value
                elif key.startswith("FC3u."):
                    inverted[key.replace("FC3u.", "FC.0.2.")] = value
                else:
                    inverted[key] = value
        else:
            inverted = state_dict.copy()
        return inverted

    def _test_bidirectional(self, new_model_class, bak_model_class, model_type, config, inputs):
        """Test that state dicts can be converted both ways."""
        # Create models
        new_model = new_model_class(config)
        bak_model = bak_model_class(config)

        # Test new -> old -> new cycle
        original_new_state = new_model.state_dict()
        translated_to_old = translate_state_dict_for_model(original_new_state, model_type)
        inverted_back_to_new = self._invert_state_dict_mapping(translated_to_old, model_type)

        # Create a new model and load the inverted state
        new_model2 = new_model_class(config)
        new_model2.load_state_dict(inverted_back_to_new, strict=False)

        # Both models should produce same output
        new_model.eval()
        new_model2.eval()

        with torch.no_grad():
            out1 = new_model(*inputs)
            out2 = new_model2(*inputs)

        torch.testing.assert_close(out1, out2, rtol=1e-5, atol=1e-7,
                                   msg=f"{model_type}: bidirectional conversion mismatch")

    def test_pi_dcon_bidirectional(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()
        self._test_bidirectional(
            self.models_darcy_new.PI_DCON,
            self.models_darcy_bak.PI_DCON,
            'pi_dcon',
            config,
            inputs
        )

    def test_pi_gano_bidirectional(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()
        self._test_bidirectional(
            self.models_darcy_new.PI_GANO,
            self.models_darcy_bak.PI_GANO,
            'pi_gano',
            config,
            inputs
        )


class TestPlateModelsGradientEquivalence(unittest.TestCase):
    """Test gradient/backpropagation equivalence for Plate models."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        bak_plate_path = os.path.join(cls.this_dir, "lib", "model_plate.bak.py")
        cls.models_plate_bak = _load_bak_module("model_plate_bak", bak_plate_path)
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_plate as models_plate_new
        cls.models_plate_new = models_plate_new

    def _make_inputs(self, B=3, M=17, N=13, M_shape=11, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype, requires_grad=True)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype, requires_grad=True)
        par = torch.randn(B, N, 4, generator=g, device=device, dtype=dtype, requires_grad=True)
        par_flag = torch.randint(0, 2, (B, N), generator=g, device=device, dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device=device, dtype=dtype, requires_grad=True)
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device=device, dtype=torch.float32)
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0
        return x_coor, y_coor, par, par_flag, shape_coor, shape_flag

    def _compare_gradients(self, new_model, bak_model, inputs_args, model_type):
        """Compare gradients between new and old plate models.

        CRITICAL: Uses separate input tensors for each model to ensure
        gradient comparison is meaningful (not comparing tensor to itself).
        """
        # Create SEPARATE input tensors for each model with same seed for reproducibility
        B, M, N, M_shape = 3, 17, 13, 11
        device, dtype = "cpu", torch.float32

        # Generate inputs for NEW model
        g_new = torch.Generator(device=device)
        g_new.manual_seed(1234)
        x_coor_new = torch.randn(B, M, generator=g_new, device=device, dtype=dtype, requires_grad=True)
        y_coor_new = torch.randn(B, M, generator=g_new, device=device, dtype=dtype, requires_grad=True)
        par_new = torch.randn(B, N, 4, generator=g_new, device=device, dtype=dtype, requires_grad=True)
        par_flag_new = torch.randint(0, 2, (B, N), generator=g_new, device=device, dtype=torch.float32)
        shape_coor_new = torch.randn(B, M_shape, 2, generator=g_new, device=device, dtype=dtype, requires_grad=True)
        shape_flag_new = torch.randint(0, 2, (B, M_shape), generator=g_new, device=device, dtype=torch.float32)
        par_flag_new[:, 0] = 1.0
        shape_flag_new[:, 0] = 1.0

        # Generate inputs for BAK model (same seed for identical values)
        g_bak = torch.Generator(device=device)
        g_bak.manual_seed(1234)
        x_coor_bak = torch.randn(B, M, generator=g_bak, device=device, dtype=dtype, requires_grad=True)
        y_coor_bak = torch.randn(B, M, generator=g_bak, device=device, dtype=dtype, requires_grad=True)
        par_bak = torch.randn(B, N, 4, generator=g_bak, device=device, dtype=dtype, requires_grad=True)
        par_flag_bak = torch.randint(0, 2, (B, N), generator=g_bak, device=device, dtype=torch.float32)
        shape_coor_bak = torch.randn(B, M_shape, 2, generator=g_bak, device=device, dtype=dtype, requires_grad=True)
        shape_flag_bak = torch.randint(0, 2, (B, M_shape), generator=g_bak, device=device, dtype=torch.float32)
        par_flag_bak[:, 0] = 1.0
        shape_flag_bak[:, 0] = 1.0

        # Translate state dict and load into bak model
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        missing, unexpected = bak_model.load_state_dict(translated_state, strict=False)

        # Report any state dict loading issues (Issue #2 fix)
        if missing:
            print(f"WARNING {model_type}: Missing keys in bak model: {missing[:5]}... (showing first 5)")
        if unexpected:
            print(f"WARNING {model_type}: Unexpected keys in state dict: {unexpected[:5]}... (showing first 5)")

        new_model.train()
        bak_model.train()

        # Forward and backward for NEW model
        out_new = new_model(x_coor_new, y_coor_new, par_new, par_flag_new, shape_coor_new, shape_flag_new)
        loss_new = out_new[0].sum() + out_new[1].sum()
        loss_new.backward()

        # Forward and backward for BAK model
        out_bak = bak_model(x_coor_bak, y_coor_bak, par_bak, par_flag_bak, shape_coor_bak, shape_flag_bak)
        loss_bak = out_bak[0].sum() + out_bak[1].sum()
        loss_bak.backward()

        # Compare DIFFERENT gradient tensors (the actual fix for Issue #1)
        torch.testing.assert_close(x_coor_new.grad, x_coor_bak.grad, rtol=1e-5, atol=1e-8,
                                   msg=f"{model_type}: x_coor gradient mismatch")
        torch.testing.assert_close(y_coor_new.grad, y_coor_bak.grad, rtol=1e-5, atol=1e-8,
                                   msg=f"{model_type}: y_coor gradient mismatch")

        # Compare parameter gradients via state dict translation
        new_grads = {}
        for name, param in new_model.named_parameters():
            if param.grad is not None:
                # Translate new parameter name to old format
                translated_key = self._translate_param_name(name, model_type)
                if translated_key:
                    new_grads[translated_key] = param.grad

        bak_grads = {}
        for name, param in bak_model.named_parameters():
            if param.grad is not None:
                bak_grads[name] = param.grad

        # Compare gradients for matching parameters
        for key in new_grads:
            if key in bak_grads:
                torch.testing.assert_close(
                    new_grads[key], bak_grads[key], rtol=1e-5, atol=1e-8,
                    msg=f"{model_type}: gradient mismatch for {key}"
                )

    def _translate_param_name(self, new_name, model_type):
        """Translate new parameter name to old format."""
        if model_type == 'pi_dcon_plate':
            # FC.0.0.* -> FC1u.*, FC.0.1.* -> FC2u.*, etc.
            if new_name.startswith("FC.0.0."):
                return new_name.replace("FC.0.0.", "FC1u.")
            elif new_name.startswith("FC.0.1."):
                return new_name.replace("FC.0.1.", "FC2u.")
            elif new_name.startswith("FC.0.2."):
                return new_name.replace("FC.0.2.", "FC3u.")
            elif new_name.startswith("FC.1.0."):
                return new_name.replace("FC.1.0.", "FC1v.")
            elif new_name.startswith("FC.1.1."):
                return new_name.replace("FC.1.1.", "FC2v.")
            elif new_name.startswith("FC.1.2."):
                return new_name.replace("FC.1.2.", "FC3v.")
        elif model_type == 'pi_gano_plate':
            # FC.0.0.* -> FC1u.*, FC.0.1.* -> FC2u.*, etc.
            if new_name.startswith("FC.0.0."):
                return new_name.replace("FC.0.0.", "FC1u.")
            elif new_name.startswith("FC.0.1."):
                return new_name.replace("FC.0.1.", "FC2u.")
            elif new_name.startswith("FC.0.2."):
                return new_name.replace("FC.0.2.", "FC3u.")
            elif new_name.startswith("FC.0.3."):
                return new_name.replace("FC.0.3.", "FC4u.")
            elif new_name.startswith("FC.1.0."):
                return new_name.replace("FC.1.0.", "FC1v.")
            elif new_name.startswith("FC.1.1."):
                return new_name.replace("FC.1.1.", "FC2v.")
            elif new_name.startswith("FC.1.2."):
                return new_name.replace("FC.1.2.", "FC3v.")
            elif new_name.startswith("FC.1.3."):
                return new_name.replace("FC.1.3.", "FC4v.")
        # For DG and other shared components, names should match
        return new_name

    def test_pi_dcon_plate_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()
        new = self.models_plate_new.PI_DCON_plate(config)
        bak = self.models_plate_bak.DCON(config)
        self._compare_gradients(new, bak, inputs, 'pi_dcon_plate')

    def test_pi_gano_plate_gradients(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        inputs = self._make_inputs()
        new = self.models_plate_new.PI_GANO_plate(config)
        bak = self.models_plate_bak.GANO(config)
        self._compare_gradients(new, bak, inputs, 'pi_gano_plate')


class TestPlateModelsMultipleConfigurations(unittest.TestCase):
    """Test equivalence across various input configurations for Plate models."""

    @classmethod
    def setUpClass(cls):
        cls.this_dir = os.path.dirname(os.path.abspath(__file__))
        bak_plate_path = os.path.join(cls.this_dir, "lib", "model_plate.bak.py")
        cls.models_plate_bak = _load_bak_module("model_plate_bak", bak_plate_path)
        import sys
        if cls.this_dir not in sys.path:
            sys.path.insert(0, cls.this_dir)
        from lib import model_plate as models_plate_new
        cls.models_plate_new = models_plate_new

    def _make_inputs(self, B=3, M=17, N=13, M_shape=11, device="cpu", dtype=torch.float32):
        g = torch.Generator(device=device)
        g.manual_seed(1234)
        x_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        y_coor = torch.randn(B, M, generator=g, device=device, dtype=dtype)
        par = torch.randn(B, N, 4, generator=g, device=device, dtype=dtype)
        par_flag = torch.randint(0, 2, (B, N), generator=g, device=device, dtype=torch.float32)
        shape_coor = torch.randn(B, M_shape, 2, generator=g, device=device, dtype=dtype)
        shape_flag = torch.randint(0, 2, (B, M_shape), generator=g, device=device, dtype=torch.float32)
        par_flag[:, 0] = 1.0
        shape_flag[:, 0] = 1.0
        return x_coor, y_coor, par, par_flag, shape_coor, shape_flag

    def _test_config(self, new_model, bak_model, model_type, inputs):
        new_state = new_model.state_dict()
        translated_state = translate_state_dict_for_model(new_state, model_type)
        bak_model.load_state_dict(translated_state, strict=False)

        new_model.eval()
        bak_model.eval()

        with torch.no_grad():
            out_new = new_model(*inputs)
            out_bak = bak_model(*inputs)

        self.assertEqual(len(out_new), 2)
        self.assertEqual(len(out_bak), 2)
        torch.testing.assert_close(out_new[0], out_bak[0], rtol=1e-5, atol=1e-7)
        torch.testing.assert_close(out_new[1], out_bak[1], rtol=1e-5, atol=1e-7)

    def test_pi_dcon_plate_varying_batch_sizes(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        for B in [1, 2, 5, 10]:
            with self.subTest(batch_size=B):
                inputs = self._make_inputs(B=B)
                new = self.models_plate_new.PI_DCON_plate(config)
                bak = self.models_plate_bak.DCON(config)
                self._test_config(new, bak, 'pi_dcon_plate', inputs)

    def test_pi_gano_plate_varying_dimensions(self):
        config = {"model": {"fc_dim": 64, "N_layer": 3, "fc_dim_branch": 32}}
        test_configs = [
            (3, 10, 5, 7),
            (3, 50, 20, 15),
            (3, 100, 30, 25),
        ]
        for B, M, N, M_shape in test_configs:
            with self.subTest(M=M, N=N, M_shape=M_shape):
                inputs = self._make_inputs(B=B, M=M, N=N, M_shape=M_shape)
                new = self.models_plate_new.PI_GANO_plate(config)
                bak = self.models_plate_bak.GANO(config)
                self._test_config(new, bak, 'pi_gano_plate', inputs)


if __name__ == "__main__":
    unittest.main(verbosity=2)
