from setuptools import setup, find_packages

setup(
    name="cellcv",  # Name of your package
    version="0.1.0",  # Version of your package
    author="jendralhxr, m172205, d191761",
    author_email="zulhaj@hiroshima-u.ac.jp",
    description="Cell counting and motility evaluation estimation on from video data",
    url="https://github.com/d191761/comot",  # URL of the package repository
    packages=find_packages(),  # Automatically find sub-packages
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires="~=3.12",  # Python version required
    install_requires=[
        "opencv-python>=4.6",
        "numpy>=1.15",
        "matplotlib>=3.4",
        "customtkinter>=5.2.2",
        "pillow>=10.2.0",
    ],
)
