import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="panns-inference", # Replace with your own username
    version="0.1.1",
    author="Qiuqiang Kong",
    author_email="qiuqiangkong@gmail.com",
    description="panns_inference: audio tagging and sound event detection inference toolbox",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qiuqiangkong/panns_inference",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['matplotlib', 'librosa', 'torchlibrosa'],
    python_requires='>=3.6',
)
