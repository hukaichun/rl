from setuptools import setup

setup(
    name = "rl",
    version = "0.0.0",
    description = "reinforcement learning using tensorflow",
    author = "hu kai chun",
    author_email = "hu.kaichun@gmail.com",
    license = "MIT",
    packages = ["rl.rl_tf", "rl.rl_tf.core",
                "rl.networks_tf", "rl.utils"],
    # install_requires=[],
    zip_safe = False
)
