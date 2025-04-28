import pymust
import numpy as np
import os

# Define output directory
output_dir = "simulated_data"
os.makedirs(output_dir, exist_ok=True)

print("Starting data simulation and reference generation script...")

# Cell 2: Define Transducer
print("Defining transducer parameters...")
param3 = pymust.utils.Param()
param3.fc = 3e6  # Center frequency [Hz]
param3.bandwidth = 70  # Fractional bandwidth [%]
param3.width = 250e-6  # Element width [m]
param3.height = 250e-6 # Element height [m]
param3.radius = np.inf # Assuming flat elements for matrix array

# Position of the elements (32x32 matrix array, pitch = 300 microns)
pitch = 300e-6
param3.pitch = pitch
xe, ye = np.meshgrid((np.arange(1, 33) - 16.5) * pitch, (np.arange(1, 33) - 16.5) * pitch, indexing='ij')
ze = np.zeros(xe.shape)
# Store element coordinates in Fortran order (column-major) as expected by some MUST functions
param3.elements = np.array([xe.flatten(order="F"), ye.flatten(order="F")])
param3.Nelements = param3.elements.shape[1]
print(f"Transducer defined: {param3.Nelements} elements.")

# Cell 3: Define Single Scatterer
print("Defining single scatterer...")
# Using (1,) shape based on successful multi-scatterer example
x = np.array([0.0])
y = np.array([0.0])
z = np.array([30e-3]) # Scatterer at 30mm depth
RC = np.array([1.0]) # Reflection coefficient
print(f"Scatterer defined at ({x[0]}, {y[0]}, {z[0]}) m with RC={RC[0]}.")

# Cell 4: Define Plane Wave Transmit Delays (along z-axis)
print("Defining transmit delays for plane wave...")
txdel3 = np.zeros((1, param3.Nelements)) # Changed shape to (1, 1024)
print("Transmit delays set to zero for all elements.")

# Set sampling frequency (required for simulation and demodulation)
param3.fs = 4 * param3.fc
print(f"Sampling frequency set to {param3.fs / 1e6} MHz.")

# Cell 5: Simulate RF Data
print("Simulating RF data using pymust.simus3...")
try:
    RF3, _ = pymust.simus3(x, y, z, RC, txdel3, param3)
    print(f"RF data simulated. Shape: {RF3.shape}")
except AttributeError:
    print("Error: pymust.simus3 not found. Please ensure PyMUST is correctly installed and supports 3D simulation.")
    exit()
except Exception as e:
    print(f"An error occurred during RF simulation: {e}")
    exit()

# Cell 6: Demodulate RF Data
print("Demodulating RF data to I/Q...")
try:
    IQ3 = pymust.rf2iq(RF3, param3.fs, param3.fc)
    print(f"I/Q data generated. Shape: {IQ3.shape}")
    # Add prints for IQ data
    print(f"IQ3 data type: {IQ3.dtype}")
    print(f"Sample IQ3 data (first 5 samples, first 5 elements):\n{IQ3[:5, :5]}")

except NameError:
    print("RF data (RF3) not available. Please ensure the simulation cell ran successfully.")
    exit()
except Exception as e:
    print(f"An error occurred during demodulation: {e}")
    exit()

# Cell 7: Define Beamforming Grid
print("Defining 3D beamforming grid...")
lambda_c = 1540 / param3.fc # Approximate wavelength
n_x, n_y, n_z = 64, 64, 128
xi, yi, zi = np.meshgrid(np.linspace(-10e-3, 10e-3, n_x), # Centered around 0
                         np.linspace(-10e-3, 10e-3, n_y), # Centered around 0
                         np.linspace(20e-3, 40e-3, n_z), # Centered around 30mm depth
                         indexing='ij')
print(f"Beamforming grid defined with shape: {xi.shape}")

# Cell 8: Beamform I/Q Data using PyMUST for reference
print("Calculating beamforming matrix using pymust.dasmtx3 and applying...")
try:
    M3 = pymust.dasmtx3(IQ3, xi, yi, zi, txdel3, param3)
    bIQ3_ref = pymust.utils.applyDasMTX(M3, IQ3, xi.shape)
    print(f"PyMUST reference beamformed I/Q data generated. Shape: {bIQ3_ref.shape}")
except NameError:
    print("I/Q data (IQ3) not available. Please ensure the demodulation cell ran successfully.")
    exit()
except AttributeError as e:
     print(f"Error: PyMUST beamforming function ({e}) not found. Please ensure PyMUST is correctly installed and supports 3D beamforming.")
     exit()
except Exception as e:
    print(f"An error occurred during PyMUST beamforming: {e}")
    exit()

# Save generated data
print(f"Saving data to '{output_dir}' directory...")
np.save(os.path.join(output_dir, 'RF3.npy'), RF3)
np.save(os.path.join(output_dir, 'xi.npy'), xi)
np.save(os.path.join(output_dir, 'yi.npy'), yi)
np.save(os.path.join(output_dir, 'zi.npy'), zi)
np.save(os.path.join(output_dir, 'txdel3.npy'), txdel3)
# Save relevant parameters from param3
param_data = {
    'fs': param3.fs,
    'fc': param3.fc,
    'elements': param3.elements,
    'Nelements': param3.Nelements,
    'pitch': param3.pitch,
    'width': param3.width,
    'height': param3.height,
    'radius': param3.radius,
    'bandwidth': param3.bandwidth
}
np.save(os.path.join(output_dir, 'param_data.npy'), param_data)
np.save(os.path.join(output_dir, 'bIQ3_ref.npy'), bIQ3_ref)

print("Data simulation and reference generation complete. Data saved.")