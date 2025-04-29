import numpy as np
import os

data_dir = "simulated_data"
param_data_path = os.path.join(data_dir, 'param_data.npy')

try:
    param_data = np.load(param_data_path, allow_pickle=True).item()
    elements = param_data['elements'] # Shape (2, num_elements) -> [x_coords, y_coords]

    print(f"Shape of 'elements' array: {elements.shape}")
    print(f"Strides of 'elements' array: {elements.strides}")
    print(f"Flags of 'elements' array: {elements.flags}")
    print("-" * 20)

    # The elements are stored as [x_coords, y_coords] with shape (2, 1024)
    # Element 500 corresponds to index 499
    element_index_to_check = 499 # For element 500

    if elements.shape[1] > element_index_to_check:
        # Assuming the elements are stored as [x_coords, y_coords]
        element_x_slice = elements[0, :]
        element_y_slice = elements[1, :]
        # Assuming z=0 for element positions based on simulate_data.py
        element_z = 0.0

        print(f"Shape of 'elements[0, :]' slice: {element_x_slice.shape}")
        print(f"Strides of 'elements[0, :]' slice: {element_x_slice.strides}")
        print(f"Flags of 'elements[0, :]' slice: {element_x_slice.flags}")
        print("-" * 20)

        print(f"Shape of 'elements[1, :]' slice: {element_y_slice.shape}")
        print(f"Strides of 'elements[1, :]' slice: {element_y_slice.strides}")
        print(f"Flags of 'elements[1, :]' slice: {element_y_slice.flags}")
        print("-" * 20)


        element_x = elements[0, element_index_to_check]
        element_y = elements[1, element_index_to_check]

        print(f"Element {element_index_to_check + 1} (index {element_index_to_check}) coordinates:")
        print(f"  x: {element_x:.6f}")
        print(f"  y: {element_y:.6f}")
        print(f"  z: {element_z:.6f}") # Based on assumption

    else:
        print(f"Error: Element index {element_index_to_check} is out of bounds for the elements array with shape {elements.shape}.")


except FileNotFoundError:
    print(f"Error: '{param_data_path}' not found. Please run simulate_data.py first.")
except Exception as e:
    print(f"An error occurred: {e}")