#!/usr/bin/env python3
"""
Demonstration of the gradient comparison bug in test_models_equivalence.py

The bug: In gradient tests, the code compares x_coor.grad to itself
instead of comparing gradients between new and backup models.
"""

import torch
import numpy as np

def test_gradient_comparison_bug():
    """Demonstrate that the gradient comparison in the test is broken."""
    print("=== Demonstrating the gradient comparison bug ===\n")
    
    # Create two different models that will produce different gradients
    class ModelA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 1)
            # Initialize with specific weights
            torch.manual_seed(42)
            self.fc.weight.data = torch.randn_like(self.fc.weight)
            
        def forward(self, x):
            return self.fc(x).sum()
    
    class ModelB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(10, 1)
            # Initialize with DIFFERENT weights
            torch.manual_seed(123)
            self.fc.weight.data = torch.randn_like(self.fc.weight)
            
        def forward(self, x):
            return self.fc(x).sum()
    
    # Create models
    model_a = ModelA()
    model_b = ModelB()
    
    # Create input (notice this is a SINGLE tensor)
    x = torch.randn(5, 10, requires_grad=True)
    
    # Forward pass through model A
    out_a = model_a(x)
    loss_a = out_a.sum()
    loss_a.backward(retain_graph=True)  # Keep graph for second backward
    
    grad_after_a = x.grad.clone() if x.grad is not None else None
    print(f"Gradient after model A backward pass: {grad_after_a}")
    
    # Forward pass through model B
    out_b = model_b(x)
    loss_b = out_b.sum()
    loss_b.backward()
    
    grad_after_b = x.grad.clone() if x.grad is not None else None
    print(f"Gradient after model B backward pass: {grad_after_b}")
    
    # The gradients should be different (accumulated)
    print(f"\nAre gradients different? {not torch.allclose(grad_after_a, grad_after_b)}")
    
    # BUG: The test code compares x.grad to itself, which always passes!
    print("\n--- What the test does (BUG) ---")
    try:
        torch.testing.assert_close(x.grad, x.grad, rtol=1e-5, atol=1e-8)
        print("✓ BUG: Comparing x.grad to itself - ALWAYS PASSES!")
    except AssertionError:
        print("✗ This would never happen when comparing a tensor to itself")
    
    # CORRECT: Should compare different tensors
    print("\n--- What the test SHOULD do (CORRECT) ---")
    try:
        # This should compare grad_a and grad_b (separate tensors)
        torch.testing.assert_close(grad_after_a, grad_after_b, rtol=1e-5, atol=1e-8)
        print("✗ CORRECT: Gradients are the same (unexpected for different models)")
    except AssertionError:
        print("✓ CORRECT: Gradients are different (expected for different models)")
    
    print("\n=== Conclusion ===")
    print("The test has a bug: it compares x_coor.grad to itself,")
    print("which always passes regardless of whether the models produce")
    print("equivalent gradients or not!")

if __name__ == "__main__":
    test_gradient_comparison_bug()
