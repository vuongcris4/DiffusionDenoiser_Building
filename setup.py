from setuptools import setup, find_packages

setup(
    name='diffusion_denoiser',
    version='0.1.0',
    description='D3PM discrete diffusion for pseudo-label denoising',
    author='Hung Nguyen',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.10',
        'mmcv-full>=1.3.0',
        'numpy',
        'tqdm',
    ],
)
