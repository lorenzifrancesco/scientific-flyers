import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import random
import os

cmap_global = "bone"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Generate interference patterns from multiple sinc functions')
    parser.add_argument('--type', choices=['phase', 'amplitude', 'both'], default='both',
                        help='Type of plot to generate: phase, amplitude, or both')
    parser.add_argument('--contour', action='store_true', help='Add contour lines to the plot')
    parser.add_argument('--centers', type=int, default=25,
                        help='Number of sinc function centers (default: 25)')
    parser.add_argument('--output', default='sinc_interference_plot.jpg', 
                        help='Output file name (default: sinc_interference_plot.jpg)')
    parser.add_argument('--resolution', type=int, default=800, 
                        help='Resolution of the plot in pixels (default: 800)')
    parser.add_argument('--scale', type=float, default=1.0,
                        help='Scale factor for sinc functions (default: 1.0)')
    parser.add_argument('--noise', type=float, default=0.01,
                        help='Noise level for hexagonal lattice positions (default: 0.3)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducible results')
    return parser.parse_args()

def create_complex_grid(xmin, xmax, ymin, ymax, points=1000):
    """Create a complex grid for evaluation"""
    x = np.linspace(xmin, xmax, points)
    y = np.linspace(ymin, ymax, points)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    return X, Y, Z

def generate_hexagonal_centers(num_centers, xmin, xmax, ymin, ymax, noise_level=0.3, seed=None):
    """Generate centers in a hexagonal lattice with positional noise"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Calculate approximate lattice spacing
    # For a hexagonal lattice, we need to estimate how many rows and columns
    area = (xmax - xmin) * (ymax - ymin)
    approx_spacing = np.sqrt(area / num_centers) * 0.9  # 0.9 factor for better packing
    
    # Hexagonal lattice vectors
    a1 = np.array([1, 0]) * approx_spacing
    a2 = np.array([0.5, np.sqrt(3)/2]) * approx_spacing
    
    centers = []
    
    # Calculate how many lattice points we need in each direction
    n_cols = int((xmax - xmin) / approx_spacing) + 1
    n_rows = int((ymax - ymin) / (approx_spacing * np.sqrt(3)/2)) + 1
    
    # Center the lattice in the domain
    x_offset = (xmin + xmax) / 2
    y_offset = (ymin + ymax) / 2
    
    # Generate hexagonal lattice points
    lattice_points = []
    for i in range(-n_rows//2, n_rows//2 + 1):
        for j in range(-n_cols//2, n_cols//2 + 1):
            # Hexagonal lattice position
            pos = i * a2 + j * a1
            x_pos = pos[0] + x_offset
            y_pos = pos[1] + y_offset
            
            # Check if point is within bounds
            if xmin + 1 < x_pos < xmax - 1 and ymin + 0.5 < y_pos < ymax - 0.5:
                lattice_points.append((x_pos, y_pos))
    
    # Sort by distance from center and take the required number
    lattice_points.sort(key=lambda p: (p[0] - x_offset)**2 + (p[1] - y_offset)**2)
    selected_points = lattice_points[:num_centers]
    
    # Add noise to each selected point
    for x_pos, y_pos in selected_points:
        # Add Gaussian noise to position
        noise_x = np.random.normal(0, noise_level * approx_spacing)
        noise_y = np.random.normal(0, noise_level * approx_spacing)
        
        noisy_x = x_pos + noise_x
        noisy_y = y_pos + noise_y
        
        # Ensure noisy position stays within bounds
        noisy_x = np.clip(noisy_x, xmin + 1, xmax - 1)
        noisy_y = np.clip(noisy_y, ymin + 0.5, ymax - 0.5)
        
        centers.append(complex(noisy_x, noisy_y))
    
    return centers, approx_spacing

def sinc_2d(z, center, scale=1.0):
    """
    Calculate 2D sinc function: sinc(|z - center|) = sin(π*scale*|z - center|) / (π*scale*|z - center|)
    """
    distance = np.abs(z - center)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(distance == 0, 1.0, np.sin(np.pi * scale * distance) / (np.pi * scale * distance))
    return result

def evaluate_sinc_interference(Z, centers, scale=1.0):
    """Evaluate the superposition of sinc functions from multiple centers"""
    result = np.zeros_like(Z, dtype=complex)
    
    for i, center in enumerate(centers):
        # Calculate sinc function for this center
        sinc_val = sinc_2d(Z, center, scale)
        
        # Add random phase to each sinc function for more interesting interference
        phase = random.uniform(0, 2*np.pi)
        result += sinc_val * np.exp(1j * phase)
    
    # Handle potential infinities and NaN
    result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
    return result

def plot_function(X, Y, Z, func_values, plot_type='both', show_contour=False, resolution=800, filename_suffix=""):
    """Create plots based on the function values"""
    # Calculate amplitude and phase
    amplitude = np.abs(func_values)
    phase = np.angle(func_values)
    
    # Define A4 dimensions (in pixels at specified resolution)
    a4_width_px = int(8.27 * resolution / 100)  # Adjusted for reasonable file size
    a4_height_px = int(11.69 * resolution / 100)
    
    # Ensure output directory exists
    os.makedirs("media/sinc", exist_ok=True)
    
    # Create plots based on plot_type
    if plot_type in ['amplitude', 'both']:
        # Create figure with A4 proportions
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        
        # Remove all axes elements and margins
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Plot without any decorations
        im = ax.pcolormesh(X, Y, amplitude, cmap=cmap_global, shading='auto')
        
        if show_contour:
            ax.contour(X, Y, amplitude, colors='white', alpha=0.3, linewidths=0.5)
        
        # Save directly as JPG
        output_file = f"media/sinc/amplitude_sinc_interference{filename_suffix}.jpg"
        plt.savefig(output_file, dpi=resolution, bbox_inches='tight', pad_inches=0, format='jpg')
        plt.close()
        
        print(f"Saved amplitude plot to {output_file}")
    
    if plot_type in ['phase', 'both']:
        # Create figure with A4 proportions
        fig = plt.figure(figsize=(8.27, 11.69), dpi=100)
        
        # Remove all axes elements and margins
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Plot without any decorations
        im = ax.pcolormesh(X, Y, phase, cmap=cmap_global, shading='auto')
        
        if show_contour:
            ax.contour(X, Y, phase, colors='white', alpha=0.3, linewidths=0.5)
        
        # Save directly as JPG
        output_file = f"media/sinc/phase_sinc_interference{filename_suffix}.jpg"
        plt.savefig(output_file, dpi=resolution, bbox_inches='tight', pad_inches=0, format='jpg')
        plt.close()
        
        print(f"Saved phase plot to {output_file}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create complex grid with higher resolution for better interference patterns
    X, Y, Z = create_complex_grid(-15, 15, -10, 10, points=1500)
    
    # Generate hexagonal lattice centers with noise
    centers, lattice_spacing = generate_hexagonal_centers(args.centers, -15, 15, -10, 10, args.noise, args.seed)
    
    print(f"Generated {len(centers)} sinc function centers in hexagonal lattice")
    print(f"Lattice spacing: {lattice_spacing:.2f}")
    print(f"Noise level: {args.noise}")
    print(f"Sample centers: {[f'({c.real:.2f}, {c.imag:.2f})' for c in centers[:5]]}..." if len(centers) > 5 else f"Centers: {[f'({c.real:.2f}, {c.imag:.2f})' for c in centers]}")
    
    # Evaluate sinc interference pattern
    func_values = evaluate_sinc_interference(Z, centers, args.scale)
    
    # Create filename suffix
    filename_suffix = f"_n{args.centers}_s{args.scale:.1f}_noise{args.noise:.1f}"
    if args.seed is not None:
        filename_suffix += f"_seed{args.seed}"
    
    # Create and save plots
    plot_function(X, Y, Z, func_values, 
                 plot_type=args.type, 
                 show_contour=args.contour, 
                 resolution=args.resolution,
                 filename_suffix=filename_suffix)
    
    print(f"Hexagonal sinc interference pattern generated with {len(centers)} centers")
    print(f"Lattice spacing: {lattice_spacing:.2f}")
    print(f"Scale factor: {args.scale}")
    print(f"Noise level: {args.noise}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")

if __name__ == "__main__":
    main()