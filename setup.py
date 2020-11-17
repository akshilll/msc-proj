import setuptools


with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="centrality_experiments",
    version="0.0.1",
    author="Akshil Patel",
    author_email="akshilpatel11@gmail.com",
    description="Experiments comparing centrality measures for option discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/akshilll/centrality_experiments",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "gym", "matplotlib", "tensorflow"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: English",
    ],
)