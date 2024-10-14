import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
import pymupdf as fitz

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
  Z = filtered_noise 
  fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=300)  # A4 size
  heatmap = ax.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='autumn', alpha=0.85)
  contours = ax.contour(X, Y, Z, colors='white', linewidths=0.5)
  ax.clabel(contours, inline=True, fontsize=0, fmt="", colors='white')
  plt.tight_layout()
  plt.savefig('media/graphics3.pdf', format='pdf')
  
def crop(name="graphics3"):
  pdf_document = "media/"+name+".pdf"
  doc = fitz.open(pdf_document)

  page = doc.load_page(0)  # Load the first (and only) page
  # Define a crop box (you can adjust this based on what portion you want)
  # Coordinates are in points (1 point = 1/72 inch)
  # Let's take a portion in the center (adjust the dimensions as needed)
  rect = fitz.Rect(200, 300, 400, 600)  # Define a rectangular area (left, top, right, bottom)

  page.set_cropbox(rect)
  cropped_pdf_filename = "media/cropped/"+name+".pdf"
  doc.save(cropped_pdf_filename)
  doc.close()

  print(f'Cropped PDF saved as {cropped_pdf_filename}')

generate3()
crop()