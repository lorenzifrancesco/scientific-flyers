import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import random
import os

cmap_global = "YlGn"

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
    parser.add_argument('--dimer_separation', type=float, default=2.0,
                        help='Separation distance between dimer pairs (default: 2.0)')
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

def generate_dimer_centers(num_centers, xmin, xmax, ymin, ymax, dimer_separation=1.0, seed=None):
    """Generate pairs of nearby centers (dimers) randomly distributed"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    centers = []
    
    # We need an even number for perfect pairs, but handle odd numbers gracefully
    num_pairs = num_centers // 2
    remaining_singles = num_centers % 2
    
    # Generate dimer pairs
    for _ in range(num_pairs):
        # Generate random center for the dimer
        center_x = random.uniform(xmin + 3, xmax - 3)
        center_y = random.uniform(ymin + 2, ymax - 2)
        
        # Generate random orientation for the dimer (angle)
        angle = random.uniform(0, 2 * np.pi)
        
        # Calculate positions of the two atoms in the dimer
        dx = dimer_separation * np.cos(angle) / 2
        dy = dimer_separation * np.sin(angle) / 2
        
        # First atom of the dimer
        atom1_x = center_x - dx
        atom1_y = center_y - dy
        centers.append(complex(atom1_x, atom1_y))
        
        # Second atom of the dimer
        atom2_x = center_x + dx
        atom2_y = center_y + dy
        centers.append(complex(atom2_x, atom2_y))
    
    # Add remaining single centers if num_centers is odd
    for _ in range(remaining_singles):
        single_x = random.uniform(xmin + 2, xmax - 2)
        single_y = random.uniform(ymin + 1, ymax - 1)
        centers.append(complex(single_x, single_y))
    
    return centers, num_pairs

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
    
    # Generate dimer centers randomly distributed
    centers, num_pairs = generate_dimer_centers(args.centers, -15, 15, -10, 10, args.dimer_separation, args.seed)
    
    print(f"Generated {len(centers)} sinc function centers as dimers")
    print(f"Number of pairs: {num_pairs}")
    print(f"Dimer separation: {args.dimer_separation}")
    if len(centers) % 2 == 1:
        print(f"Plus 1 single center")
    print(f"Sample centers: {[f'({c.real:.2f}, {c.imag:.2f})' for c in centers[:6]]}..." if len(centers) > 6 else f"Centers: {[f'({c.real:.2f}, {c.imag:.2f})' for c in centers]}")
    
    # Evaluate sinc interference pattern
    func_values = evaluate_sinc_interference(Z, centers, args.scale)
    
    # Create filename suffix
    filename_suffix = f"_n{args.centers}_s{args.scale:.1f}_dimer{args.dimer_separation:.1f}"
    if args.seed is not None:
        filename_suffix += f"_seed{args.seed}"
    
    # Create and save plots
    plot_function(X, Y, Z, func_values, 
                 plot_type=args.type, 
                 show_contour=args.contour, 
                 resolution=args.resolution,
                 filename_suffix=filename_suffix)
    
    print(f"Dimer sinc interference pattern generated with {len(centers)} centers")
    print(f"Number of pairs: {num_pairs}")
    print(f"Dimer separation: {args.dimer_separation}")
    print(f"Scale factor: {args.scale}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")

if __name__ == "__main__":
    main()