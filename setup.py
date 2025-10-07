from setuptools import setup, find_packages

setup(
    name="ai-interview-app",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "setuptools>=70.0.0",
        "wheel>=0.43.0",
        "numpy==2.0.1",
        "Flask==3.0.3",
        "Werkzeug==3.0.3",
        # Add all other packages from requirements.txt
    ],
    python_requires=">=3.12",
)