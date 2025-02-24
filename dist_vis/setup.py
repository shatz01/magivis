from setuptools import setup, find_packages

setup(
    name="dist_vis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",  # Since we use torch.distributed
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A visualization tool for PyTorch distributed operations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dist_vis",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 