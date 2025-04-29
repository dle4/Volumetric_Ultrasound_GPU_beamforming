import setuptools
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Define the CUDA extension
fused_beamform_extension = CUDAExtension(
    name='fused_beamform_ext', # Name of the Python module
    sources=[
        'fused_kernel_ext/fused_beamform_wrapper.cpp',
        'fused_kernel_ext/fused_beamform_kernel.cu',
    ],
    # Add any extra compilation flags if needed
    extra_compile_args={'cxx': [], 'nvcc': []}
)

# Setup function
setuptools.setup(
    name='fused_beamform_ext',
    version='0.1',
    author='Your Name', # Replace with your name
    author_email='your.email@example.com', # Replace with your email
    description='A PyTorch C++ extension for fused beamforming kernel',
    long_description='',
    long_description_content_type='text/markdown',
    url='http://example.com/your_package', # Replace with your package URL
    packages=setuptools.find_packages(),
    ext_modules=[fused_beamform_extension],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)