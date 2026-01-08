#!/usr/bin/env python3
"""
Demonstration of why strict=False in load_state_dict can hide issues
"""

import torch

def test_strict_false_issue():
    """Demonstrate how strict=False can hide state dict mismatches."""
    print("=== Demonstrating strict=False issue ===\n")
    
    # Model A - has parameters layer1 and layer2
    class ModelA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = torch.nn.Linear(10, 5)
            self.layer2 = torch.nn.Linear(5, 1)
    
    # Model B - has different parameter names
    class ModelB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 5)
            self.fc2 = torch.nn.Linear(5, 1)
    
    model_a = ModelA()
    model_b = ModelB()
    
    # Get state dicts
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    
    print("Model A state dict keys:", list(state_a.keys()))
    print("Model B state dict keys:", list(state_b.keys()))
    print()
    
    # Try loading with strict=True (should fail)
    print("--- Loading with strict=True ---")
    try:
        model_b.load_state_dict(state_a, strict=True)
        print("✗ Unexpectedly succeeded")
    except RuntimeError as e:
        print(f"✓ Expected error: {str(e)[:100]}...")
    
    print()
    
    # Try loading with strict=False (will "silently" fail)
    print("--- Loading with strict=False ---")
    try:
        missing, unexpected = model_b.load_state_dict(state_a, strict=False)
        print("✓ Load 'succeeded' but:")
        print(f"  - Missing keys: {missing}")
        print(f"  - Unexpected keys: {unexpected}")
        print(f"  - Model B parameters were NOT actually updated!")
    except RuntimeError as e:
        print(f"✗ Unexpected error: {e}")
    
    print()
    
    # Show what parameters actually contain
    print("--- Parameter values after load ---")
    print(f"model_b.fc1.weight (should be unchanged): {model_b.fc1.weight.data[0, 0]:.6f}")
    print(f"model_b.fc2.weight (should be unchanged): {model_b.fc2.weight.data[0, 0]:.6f}")
    
    print()
    print("=== Conclusion ===")
    print("Using strict=False allows the test to continue even when")
    print("state dict keys don't match, potentially hiding bugs!")

if __name__ == "__main__":
    test_strict_false_issue()
