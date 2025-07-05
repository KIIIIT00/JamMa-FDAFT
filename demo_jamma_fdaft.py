"""
JamMa-FDAFT Complete Demonstration Script

This script demonstrates the integrated JamMa-FDAFT pipeline for planetary remote sensing image matching.
It creates synthetic planetary images, processes them through FDAFT feature extraction,
applies Joint Mamba for feature interaction, and performs hierarchical coarse-to-fine matching.

Usage:
    python demo_jamma_fdaft.py

Features demonstrated:
- FDAFT: Double-frequency scale space and robust feature extraction
- Joint Mamba (JEGO): Efficient long-range feature interaction  
- C2F Matching: Hierarchical coarse-to-fine matching with sub-pixel refinement
- Complete end-to-end pipeline optimized for planetary images
"""

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from jamma_fdaft.backbone_fdaft import FDAFTEncoder
    from jamma_fdaft.jamma_fdaft import JamMaFDAFT
    from lightning.lightning_jamma_fdaft import PL_JamMaFDAFT
    from config.default import get_cfg_defaults
    from utils.plotting import make_matching_figures
except ImportError as e:
    print(f"Error importing JamMa-FDAFT modules: {e}")
    print("Please ensure the project is properly set up.")
    sys.exit(1)


def create_planetary_image_pair():
    """
    Create a pair of synthetic planetary images for demonstration
    
    Returns:
        tuple: (image1, image2) - pair of synthetic planetary surface images
    """
    print("  Creating synthetic planetary surface images...")
    np.random.seed(42)
    size = (512, 512)
    
    # Create base terrain using multiple frequency components
    x, y = np.meshgrid(np.linspace(0, 10, size[1]), np.linspace(0, 10, size[0]))
    
    # Multi-scale terrain generation
    terrain1 = (
        np.sin(x) * np.cos(y) +                    # Large-scale features
        0.5 * np.sin(2*x) * np.cos(3*y) +         # Medium-scale features  
        0.3 * np.sin(5*x) * np.cos(2*y) +         # Small-scale features
        0.2 * np.sin(8*x) * np.cos(5*y)           # Fine details
    )
    
    # Add realistic noise for surface texture
    noise1 = np.random.normal(0, 0.1, size)
    image1 = terrain1 + noise1
    
    # Add crater-like circular depressions
    crater_positions = [
        (128, 150, 25),  # (center_x, center_y, radius)
        (300, 200, 35),
        (400, 400, 20),
        (150, 350, 30)
    ]
    
    for cx, cy, radius in crater_positions:
        y_coords, x_coords = np.ogrid[:size[0], :size[1]]
        crater_mask = (x_coords - cx)**2 + (y_coords - cy)**2 <= radius**2
        
        # Create realistic crater profile (Gaussian depression)
        distance = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
        crater_depth = np.exp(-distance**2 / (2 * (radius/2)**2)) * 0.4
        
        # Apply crater effect
        image1[crater_mask] -= crater_depth[crater_mask]
    
    # Create second image with geometric transformation
    center = (size[1]//2, size[0]//2)
    angle = 12  # degrees
    scale = 0.95
    
    # Apply transformation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += 25  # translation x
    M[1, 2] += 15  # translation y
    
    image2 = cv2.warpAffine(image1, M, (size[1], size[0]))
    
    # Simulate illumination differences
    illumination_gradient_x = np.linspace(0.85, 1.15, size[1])
    illumination_gradient_y = np.linspace(1.05, 0.95, size[0])
    illumination_map = np.outer(illumination_gradient_y, illumination_gradient_x)
    
    image2 = image2 * illumination_map
    
    # Add different noise pattern
    noise2 = np.random.normal(0, 0.08, size)
    image2 = image2 + noise2 + 0.1  # brightness change
    
    # Normalize both images to [0, 255] range
    image1 = ((image1 - image1.min()) / (image1.max() - image1.min()) * 255).astype(np.uint8)
    image2 = ((image2 - image2.min()) / (image2.max() - image2.min()) * 255).astype(np.uint8)
    
    print("  ✓ Synthetic planetary images created successfully!")
    return image1, image2


def prepare_data_batch(image1, image2):
    """
    Prepare data batch for JamMa-FDAFT processing
    
    Args:
        image1, image2: Input images
        
    Returns:
        Dictionary containing properly formatted data for the model
    """
    # Convert to RGB format (3 channels)
    if len(image1.shape) == 2:
        image1_rgb = np.stack([image1, image1, image1], axis=2)
        image2_rgb = np.stack([image2, image2, image2], axis=2)
    else:
        image1_rgb = image1
        image2_rgb = image2
    
    # Convert to torch tensors and normalize
    image1_tensor = torch.from_numpy(image1_rgb).float().permute(2, 0, 1) / 255.0
    image2_tensor = torch.from_numpy(image2_rgb).float().permute(2, 0, 1) / 255.0
    
    # Add batch dimension
    image1_batch = image1_tensor.unsqueeze(0)  # [1, 3, H, W]
    image2_batch = image2_tensor.unsqueeze(0)  # [1, 3, H, W]
    
    # Create data dictionary
    data = {
        'imagec_0': image1_batch,
        'imagec_1': image2_batch,
        'dataset_name': ['JamMa-FDAFT-Demo'],
        'scene_id': 'demo_scene',
        'pair_id': 0,
        'pair_names': [('demo_image1.png', 'demo_image2.png')]
    }
    
    return data


def demonstrate_jamma_fdaft():
    """Main demonstration function"""
    print("JamMa-FDAFT Integrated Pipeline Demonstration")
    print("=" * 60)
    print("Architecture: Input Images → FDAFT Encoder → Joint Mamba (JEGO) → C2F Matching")
    print()
    
    # Step 1: Create sample images
    print("Step 1: Creating synthetic planetary images...")
    start_time = time.time()
    image1, image2 = create_planetary_image_pair()
    creation_time = time.time() - start_time
    print(f"  ✓ Images created in {creation_time:.2f} seconds")
    
    # Display input images
    print("\nDisplaying input images...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.imshow(image1, cmap='gray')
    ax1.set_title('Planetary Image 1 (Reference)', fontsize=14, fontweight='bold')
    ax1.axis('off')
    
    ax2.imshow(image2, cmap='gray')
    ax2.set_title('Planetary Image 2 (Transformed)', fontsize=14, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Step 2: Initialize JamMa-FDAFT
    print("\nStep 2: Initializing JamMa-FDAFT model...")
    
    # Get default configuration
    config = get_cfg_defaults()
    
    # Configure for demo (smaller model for faster processing)
    config.FDAFT.NUM_LAYERS = 2
    config.FDAFT.MAX_KEYPOINTS = 500
    config.JAMMA.COARSE.D_MODEL = 128  # Smaller for demo
    config.JAMMA.FINE.D_MODEL = 32
    
    # Initialize model components
    print("  Initializing FDAFT Encoder...")
    fdaft_encoder = FDAFTEncoder(
        num_layers=config.FDAFT.NUM_LAYERS,
        sigma_0=config.FDAFT.SIGMA_0,
        use_structured_forests=config.FDAFT.USE_STRUCTURED_FORESTS,
        max_keypoints=config.FDAFT.MAX_KEYPOINTS,
        nms_radius=config.FDAFT.NMS_RADIUS
    )
    
    print("  Initializing JamMa-FDAFT Matcher...")
    jamma_fdaft = JamMaFDAFT(config=config.JAMMA)
    
    print("  ✓ JamMa-FDAFT initialized successfully!")
    
    # Print model information
    print("\nModel Architecture Information:")
    fdaft_info = fdaft_encoder.get_fdaft_info()
    matcher_info = jamma_fdaft.get_model_info()
    
    print(f"  FDAFT Encoder: {fdaft_info['encoder_type']}")
    print(f"    - Scale space layers: {fdaft_info['scale_space_layers']}")
    print(f"    - Structured Forests: {fdaft_info['structured_forests']}")
    print(f"  JamMa Matcher: {matcher_info['model_name']}")
    print(f"    - Architecture: {matcher_info['architecture']}")
    
    # Step 3: Prepare data and run inference
    print("\nStep 3: Running JamMa-FDAFT pipeline...")
    start_time = time.time()
    
    try:
        # Prepare data batch
        data = prepare_data_batch(image1, image2)
        
        # Set models to evaluation mode
        fdaft_encoder.eval()
        jamma_fdaft.eval()
        
        with torch.no_grad():
            print("  Processing through FDAFT Encoder...")
            # Extract FDAFT features
            fdaft_encoder(data)
            
            print("  Processing through Joint Mamba and C2F Matching...")
            # Apply JamMa-FDAFT matching
            jamma_fdaft(data, mode='test')
        
        processing_time = time.time() - start_time
        print(f"  ✓ Pipeline completed in {processing_time:.2f} seconds")
        
    except Exception as e:
        print(f"  ✗ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Analyze results
    print("\nStep 4: Analyzing results...")
    
    # Extract matching results
    num_matches = data.get('num_final_matches', 0)
    coarse_matches = len(data.get('mkpts0_c', []))
    fine_matches = len(data.get('mkpts0_f', []))
    
    print(f"  Coarse-level matches found: {coarse_matches}")
    print(f"  Fine-level matches found: {fine_matches}")
    print(f"  Final matches after RANSAC: {num_matches}")
    
    # Step 5: Visualize results
    print(f"\nStep 5: Visualizing results...")
    try:
        if num_matches > 0:
            # Create matching visualization
            make_matching_figures(data, mode='evaluation')
            print("  ✓ Visualization completed")
        else:
            print("  ⚠ No matches found for visualization")
            
    except Exception as e:
        print(f"  ✗ Visualization error: {e}")
    
    # Final summary
    print("\n" + "="*60)
    print("JAMMA-FDAFT DEMONSTRATION SUMMARY")
    print("="*60)
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Final matches found: {num_matches}")
    
    if num_matches >= 8:
        print("✓ SUCCESS: JamMa-FDAFT successfully matched the planetary images!")
        print("  The integrated pipeline demonstrated:")
        print("  - FDAFT: Robust feature extraction for weak surface textures")
        print("  - Joint Mamba: Efficient long-range feature interaction")
        print("  - C2F Matching: Hierarchical matching with sub-pixel refinement")
        print("  - Planetary optimization: Enhanced performance on challenging surfaces")
    else:
        print("⚠ LIMITED SUCCESS: Few matches found.")
        print("  This may be due to:")
        print("  - Reduced model size for demo (use full model for better results)")
        print("  - Challenging synthetic image characteristics")
        print("  - Need for parameter tuning for specific image types")
    
    print(f"\nNext steps:")
    print(f"  - Train on real planetary datasets (Mars, Moon, etc.)")
    print(f"  - Run: python train.py configs/data/megadepth_trainval_832.py configs/jamma_fdaft/outdoor/final.py")
    print(f"  - Test: python test.py configs/data/megadepth_test_1500.py configs/jamma_fdaft/outdoor/test.py")
    
    return True


if __name__ == "__main__":
    """Entry point for the demonstration script"""
    try:
        # Set matplotlib backend for environments without display
        try:
            import matplotlib
            matplotlib.use('TkAgg')
        except:
            matplotlib.use('Agg')
            print("Note: Using non-interactive matplotlib backend")
        
        success = demonstrate_jamma_fdaft()
        
        if success:
            print(f"\n🎉 JamMa-FDAFT demo completed successfully!")
            input("Press Enter to exit...")
        else:
            print(f"\n❌ Demo failed. Please check the error messages above.")
            
    except KeyboardInterrupt:
        print(f"\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
    sys.exit(0)
