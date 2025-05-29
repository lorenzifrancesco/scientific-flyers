import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
import pymupdf as fitz
import fitz  # PyMuPDF
from PIL import Image
import os

def generate3():
  np.random.seed(42)
  noise = np.random.randn(2000, 2000)
  F_noise = fft2(noise)
  F_noise_shifted = fftshift(F_noise)  # Shift the zero frequency to the center

  X, Y = np.meshgrid(np.linspace(-1, 1, noise.shape[0]), np.linspace(-1, 1, noise.shape[1]))
  alpha = 7
  distance = np.sqrt(X**2 + Y**2)
  filter_mask = np.exp(-(distance * 100)**2)  # Gaussian low-pass filter
  F_noise_filtered = F_noise_shifted * filter_mask
  filtered_noise = np.real(ifft2(fftshift(F_noise_filtered)))

  max = np.max(filtered_noise)
  min = np.min(filtered_noise)

  Z1 = max * np.sin(X**2 + Y**2 * 50)  # Radial pattern
  # Z2 = np.cos(15 * X) * np.sin(15 * Y)  # Wave-like stripes
  # Elliptic-hyperbolic coordinates transformation
  a = 2
  b = 0.5
  
  # r = np.sqrt(X**2 + Y**2)
  # mu = np.arccosh(r / a)
  # sinh_mu = np.sinh(mu)
  # cosh_mu = np.cosh(mu)
  # nu = np.arctan2(Y / sinh_mu, X / cosh_mu)
  mu = np.sqrt((X**2 / a**2) + (Y**2 / b**2))  # Elliptic coordinate
  nu = np.arctan2(Y, X)                        # Hyperbolic coordinate

  # Combine filtered noise with coordinate-based patterns
  Z = np.sin(mu * 10) * np.cos(nu * 5)
  # Z = filtered_noise 
  fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)  # A4 size
  heatmap = ax.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='PuBuGn', alpha=0.85)
  contours = ax.contour(X, Y, Z, colors='white', linewidths=0.5)
  # ax.clabel(contours, inline=True, fontsize=0, fmt="", colors='white')
  plt.tight_layout()
  plt.savefig('media/graphics3.png', format='png')
  
# def crop(name="graphics3"):
#   pdf_document = "media/"+name+".pdf"
#   doc = fitz.open(pdf_document)

#   page = doc.load_page(0)  # Load the first (and only) page
#   # Define a crop box (you can adjust this based on what portion you want)
#   # Coordinates are in points (1 point = 1/72 inch)
#   # Let's take a portion in the center (adjust the dimensions as needed)
#   rect = fitz.Rect(200, 300, 400, 600)  # Define a rectangular area (left, top, right, bottom)

#   page.set_cropbox(rect)
#   cropped_pdf_filename = "media/cropped/"+name+".pdf"
#   doc.save(cropped_pdf_filename)
#   doc.close()

#   print(f'Cropped PDF saved as {cropped_pdf_filename}')


def crop(name="graphics3", file_type="png"):
    if file_type.lower() == "pdf":
        # Handle PDF cropping
        pdf_document = f"media/{name}.pdf"
        doc = fitz.open(pdf_document)

        page = doc.load_page(0)  # Load the first (and only) page
        # Define a crop box (adjust this based on desired portion)
        rect = fitz.Rect(200, 300, 400, 600)  # Define rectangular area (left, top, right, bottom)

        page.set_cropbox(rect)
        cropped_pdf_filename = f"media/cropped/{name}.pdf"
        doc.save(cropped_pdf_filename)
        doc.close()

        print(f'Cropped PDF saved as {cropped_pdf_filename}')
    
    elif file_type.lower() == "png":
        # Handle PNG cropping
        png_document = f"media/{name}.png"
        
        # Check if the file exists to avoid errors
        if os.path.exists(png_document):
            image = Image.open(png_document)
            # Define crop box (adjust this based on desired portion)
            crop_box = (200, 300, 400, 600)  # (left, upper, right, lower)
            
            cropped_image = image.crop(crop_box)
            cropped_png_filename = f"media/cropped/{name}.png"
            cropped_image.save(cropped_png_filename)

            print(f'Cropped PNG saved as {cropped_png_filename}')
        else:
            print(f"{png_document} not found.")
    else:
        print("Unsupported file type. Use 'pdf' or 'png'.")

generate3()
crop()