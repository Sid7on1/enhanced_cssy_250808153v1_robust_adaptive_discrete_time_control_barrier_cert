import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="enhanced_cs_sy_2508_agent",
    version="1.0.0",
    author="XR Eye Tracking Team",
    author_email="xyzt@example.com",
    description="Enterprise XR Eye Tracking System - Agent Component",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://example.com/xr-eye-tracking",
    project_urls={
        "Bug Reports": "https://example.com/xr-eye-tracking/issues",
        "Funding": "https://example.com/xr-eye-tracking/backers",
        "Say Thanks!": "https://example.com/xr-eye-tracking/sponsors",
        "Contact": "https://example.com/xr-eye-tracking/contact",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch==1.13.1",
        "numpy==1.24.2",
        "pandas==1.5.2",
        "scipy==1.9.3",
        "matplotlib==4.5.0",
        "control==0.9.4",  # Control systems library
        "llvmlite==0.39.0",
        "numba==0.58.0",
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        "enhanced_cs_sy_2508_agent": ["data/models/*", "data/config.ini"]
    },
    entry_points={
        "console_scripts": [
            "enhanced_cs_sy_2508_agent = enhanced_cs_sy_2508_agent.__main__:main"
        ]
    },
)

This setup.py script is for a Python package named "enhanced_cs_sy_2508_agent", which is a component of the Enterprise XR Eye Tracking System project. It includes the necessary metadata and dependencies for the package.

The script sets up the package installation using the setuptools module. It specifies the package name, version, author, description, and other relevant information. The long_description is read from a separate README.md file.

The install_requires section lists the required dependencies, including specific versions of torch, numpy, pandas, and other libraries. These dependencies will be installed automatically when the package is installed.

The packages parameter discovers and includes all packages under the current directory. The include_package_data option ensures that any package data (such as model files or configuration files) are included in the distribution.

Additionally, the entry_points section specifies a console script that serves as an entry point for the package, allowing it to be executed directly from the command line.

Overall, this setup.py script provides the necessary information for distributing and installing the "enhanced_cs_sy_2508_agent" package, making it compatible with the larger XR Eye Tracking System project.