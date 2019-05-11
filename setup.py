from setuptools import setup

setup(
    name = "rl",
    version = "0.0.0",
    description = "reinforcement learning using tensorflow",
    author = "hu kai chun",
    author_email = "hu.kaichun@gmail.com",
    license = "MIT",
    packages = ["rl.core", "rl.networks_tf", "rl.alg"],
    install_requires=["tensorflow-probability"],
    zip_safe = False
)
