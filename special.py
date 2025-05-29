import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import special
import argparse
import random
from PIL import Image
import os

cmap_global = "gnuplot2"

def parse_arguments():
    parser = argparse.ArgumentParser(description='Plot special functions on the complex plane')
    parser.add_argument('--type', choices=['phase', 'amplitude', 'both'], default='both',
                        help='Type of plot to generate: phase, amplitude, or both')
    parser.add_argument('--contour', action='store_true', help='Add contour lines to the plot')
    parser.add_argument('--function', choices=['gamma', 'bessel', 'airy', 'zeta', 
                                               'legendre', 'hankel', 'spherical_bessel', 
                                               'random'], default='random',
                        help='Special function to plot (default: random)')
    parser.add_argument('--output', default='special_function_plot.jpg', 
                        help='Output file name (default: special_function_plot.jpg)')
    parser.add_argument('--resolution', type=int, default=800, 
                        help='Resolution of the plot in pixels (default: 800)')
    return parser.parse_args()

def create_complex_grid(xmin, xmax, ymin, ymax, points=500):
    """Create a complex grid for evaluation"""
    x = np.linspace(xmin, xmax, points)
    y = np.linspace(ymin, ymax, points)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y
    return X, Y, Z

def evaluate_function(Z, func_name):
    """Evaluate the selected special function on the complex grid"""
    # Dictionary of functions with their evaluation methods and display names
    functions = {
        'gamma':            (lambda z: special.gamma(z), "Gamma Function Γ(z)"),
        'bessel':           (lambda z: special.jv(1, z), "Bessel Function J₁(z)"),
        'airy':             (lambda z: special.airy(z)[0], "Airy Function Ai(z)"),
        'zeta':             (lambda z: special.zeta(z), "Riemann Zeta Function ζ(z)"),
        'legendre':         (lambda z: special.lpmv(0, 1, z), "Legendre Polynomial P₁(z)"),
        'hankel':           (lambda z: special.hankel1(0, z), "Hankel Function H₀⁽¹⁾(z)"),
        'spherical_bessel': (lambda z: special.spherical_jn(1, z), "Spherical Bessel Function j₁(z)")
    }
    
    if func_name == 'random':
        func_name = random.choice(list(functions.keys()))
    
    func, func_display_name = functions[func_name]
    
    # Handle potential warnings and errors during function evaluation
    with np.errstate(all='ignore'):
        try:
            result = func(Z)
            # Replace infinities and NaN with a very large finite number for plotting
            result = np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10)
            return result, func_name, func_display_name
        except Exception as e:
            print(f"Error evaluating function {func_name}: {e}")
            # Return a default value in case of error
            return np.ones_like(Z), func_name, f"{func_display_name} (evaluation error)"

def plot_function(X, Y, Z, func_values, func_name, func_display_name, plot_type='both', show_contour=False, resolution=800):
    """Create plots based on the function values"""
    # Calculate amplitude and phase
    amplitude = np.abs(func_values)
    phase = np.angle(func_values)
    
    # Define A4 dimensions (in pixels at specified resolution)
    a4_width_px = int(8.27 * resolution)
    a4_height_px = int(11.69 * resolution)
    
    # Create plots based on plot_type
    if plot_type in ['amplitude', 'both']:
        # Create figure with exact A4 dimensions
        fig = plt.figure(figsize=(8.27, 11.69), dpi=resolution)
        
        # Remove all axes elements and margins
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Plot without any decorations
        im = ax.pcolormesh(X, Y, amplitude, cmap=cmap_global, shading='auto')
        
        if show_contour:
            ax.contour(X, Y, amplitude, colors='white', alpha=0.5, linewidths=0.5)
        
        # Save directly as JPG (no temporary PNG)
        output_file = f"media/special/amplitude_{func_name}.jpg"
        plt.savefig(output_file, dpi=resolution, bbox_inches='tight', pad_inches=0, format='jpg')
        plt.close()
        
        # Ensure exact A4 dimensions by resizing if needed
        with Image.open(output_file) as img:
            img = img.convert('RGB')
            img = img.resize((a4_width_px, a4_height_px), Image.LANCZOS)
            img.save(output_file, quality=95)
        
        print(f"Saved amplitude plot to {output_file}")
    
    if plot_type in ['phase', 'both']:
        # Create figure with exact A4 dimensions
        fig = plt.figure(figsize=(8.27, 11.69), dpi=resolution)
        
        # Remove all axes elements and margins
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Plot without any decorations
        im = ax.pcolormesh(X, Y, phase, cmap=cmap_global, shading='auto')
        
        if show_contour:
            ax.contour(X, Y, phase, colors='white', alpha=0.5, linewidths=0.5)
        
        # Save directly as JPG (no temporary PNG)
        output_file = f"media/special/phase_{func_name}.jpg"
        plt.savefig(output_file, dpi=resolution, bbox_inches='tight', pad_inches=0, format='jpg')
        plt.close()
        
        # Ensure exact A4 dimensions by resizing if needed
        with Image.open(output_file) as img:
            img = img.convert('RGB')
            img = img.resize((a4_width_px, a4_height_px), Image.LANCZOS)
            img.save(output_file, quality=95)
        
        print(f"Saved phase plot to {output_file}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Create complex grid
    X, Y, Z = create_complex_grid(-20, 20, -6, 6, points=2000)
    
    # Evaluate function
    func_values, selected_func, func_display_name = evaluate_function(Z, args.function)
    
    # Create and save plots
    plot_function(X, Y, Z, func_values, selected_func, func_display_name,
                 plot_type=args.type, show_contour=args.contour, resolution=args.resolution)
    
    print(f"Function plotted: {func_display_name}")

if __name__ == "__main__":
    main()