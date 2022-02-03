import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="poreana",
    version="0.2.2",
    author="Hamzeh Kraus",
    author_email="kraus@itt.uni-stuttgart.de",
    description="Pore system analysis tool.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Ajax23/PoreAna",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
    install_requires=['pandas', 'seaborn', 'scipy', 'chemfiles==0.10.0', 'porems>=0.2.5', 'jinja2', 'h5py', 'pyyaml'],
    include_package_data=True,
)
