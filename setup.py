from setuptools import setup

setup(
    name="adversarial_gym",
    version="0.0.1dev",
    description="Simple utilities to add noise to RL environments.",
    install_requires=[
        "gym",
        "numpy"
    ]
)