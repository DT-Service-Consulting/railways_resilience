from setuptools import setup, find_packages

setup(
    name="railways_resilience",
    version="1.0.0",
    author="Marco Di Gennaro",
    author_email="marco.digennaro@dtsc.be",
    description="Public Transport Network Graph Modelling using GTFS Data",
    long_description=open("../README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DT-Service-Consulting/railways_resilience",  # Replace with your repository URL
    packages=find_packages(exclude=["notebooks", "tests"]),
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "networkx",
        "matplotlib",
        "jupyter",
        "gtfspy",  # Assuming this is a custom library
        "osmread",  # Assuming this is a custom library
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "gtfs-network=src.main:main",  # Replace `src.main:main` with your main script entry point
        ],
    },
)