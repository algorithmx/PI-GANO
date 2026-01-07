# PI-GANO Model Equivalence Testing

## Overview

The `test_models_equivalence.py` file provides comprehensive testing to verify that the refactored models in `lib/model_darcy.py` and `lib/model_plate.py` are functionally equivalent to the original `.bak` implementations.

## Test Structure

### Test Classes

1. **TestDarcyModelsForwardEquivalence**
   - Tests forward pass equivalence for all Darcy flow models
   - Models tested:
     - `PI_DCON` vs original
     - `PI_PN` vs original
     - `PI_PN_only_geo` vs original
     - `PI_GANO` vs original
     - `PI_GANO_add` vs original
     - `PI_GANO_mul` vs original

2. **TestPlateModelsForwardEquivalence**
   - Tests forward pass equivalence for all plate stress models (2-component vector fields)
   - Models tested:
     - `PI_DCON_plate` vs original
     - `PI_PN_plate` vs original
     - `PI_PN_only_geo_plate` vs original
     - `PI_GANO_plate` vs original
     - `PI_GANO_add_plate` vs original
     - `PI_GANO_mul_plate` vs original

3. **TestDarcyModelsParameterEquivalence**
   - Verifies parameter counts match exactly for all Darcy models

4. **TestPlateModelsParameterEquivalence**
   - Verifies parameter counts match exactly for all plate models

## State Dict Mapping

The test includes comprehensive state dict mapping functions to handle parameter name changes between refactored and original implementations:

### Darcy Models
- `map_pi_dcon_state_dict`: `FC.0.*` → `FC1u.*`, `FC.1.*` → `FC2u.*`, `FC.2.*` → `FC3u.*`
- `map_pi_gano_state_dict`: Maps FC layers and DG encoder
- `map_pi_gano_add_state_dict` / `map_pi_gano_mul_state_dict`: Handles 2*fc_dim variants

### Plate Models
- `map_pi_dcon_plate_state_dict`: Maps both u and v component FC layers
- `map_pi_gano_plate_state_dict`: Maps 4-layer trunk networks for both components

## Running the Tests

```bash
cd /Users/dabajabaza/Documents/Nutstore/Work/Project/DeepONet/PI-GANO
python test_models_equivalence.py
```

### Expected Output

The tests will output detailed results for each test case:

```
test_pi_dcon_forward (__main__.TestDarcyModelsForwardEquivalence) ... ok
test_pi_pn_forward (__main__.TestDarcyModelsForwardEquivalence) ... ok
test_pi_gano_forward (__main__.TestDarcyModelsForwardEquivalence) ... ok
...
test_pi_dcon_parameters (__main__.TestDarcyModelsParameterEquivalence) ... ok
...
```

## Test Coverage

✅ **Forward Pass Equivalence**:
   - Random inputs with fixed seeds for reproducibility
   - State dict translation and loading
   - Numerical precision verification (rtol=1e-5, atol=1e-7)

✅ **Parameter Count Verification**:
   - Total parameters match exactly
   - All parameters are trainable
   - Layer-by-layer breakdown comparison

## Key Features

1. **Automatic State Dict Translation**: Tests automatically translate parameter names between new and old formats
2. **Comprehensive Coverage**: All major model variants are tested
3. **Numerical Precision**: Uses `torch.testing.assert_close` with strict tolerances
4. **Modular Design**: Easy to extend with new model types

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- unittest (included in Python standard library)

## Notes

- The test file follows the same structure as `PI-DCON/Main/test_models_equivalence.py`
- Tests are positioned in the project root directory for easy access
- Backup models are loaded from `lib/*.bak.py` files
- New models are imported from `lib/model_darcy.py` and `lib/model_plate.py`

## Future Enhancements

Potential additions to the test suite:
- Gradient computation equivalence tests
- Backward pass verification
- Performance benchmarks
- Memory usage comparison
- Test with different input sizes and batch dimensions
