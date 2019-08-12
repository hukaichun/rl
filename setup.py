from setuptools import setup

setup(
    name = "ReinforementLearning",
    version = "0.0.1",
    description = "reinforcement learning using tensorflow 2.0",
    author = "hu kai chun",
    author_email = "hu.kaichun@gmail.com",
    license = "",
    packages = ["ReinforcementLearning",
                "ReinforcementLearning.util"],
    install_requires=["tensorflow-probability"],
    zip_safe = False
)
