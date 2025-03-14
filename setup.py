# Lint as: python3
"""
Neural-Forge-Ai
"""
import os
from setuptools import find_packages, setup

DOCLINES = __doc__.split("\n")

this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# get INSTALL_REQUIRES from requirements.txt (if it exists)
INSTALL_REQUIRES = []
requirements_path = os.path.join(this_directory, "requirements.txt")
if os.path.exists(requirements_path):
    with open(requirements_path, encoding="utf-8") as f:
        for line in f:
            # Exclude 'bitsandbytes' if installing on macOS
            if "bitsandbytes" in line:
                line = line.strip() + " ; sys_platform == 'linux'"
                INSTALL_REQUIRES.append(line.strip())
            else:
                INSTALL_REQUIRES.append(line.strip())

QUALITY_REQUIRE = [
    "black",
    "isort",
    "flake8==3.7.9",
]

TESTS_REQUIRE = ["pytest"]

CLIENT_REQUIRES = ["requests", "loguru"]

EXTRAS_REQUIRE = {
    "base": INSTALL_REQUIRES,
    "dev": INSTALL_REQUIRES + QUALITY_REQUIRE + TESTS_REQUIRE,
    "quality": INSTALL_REQUIRES + QUALITY_REQUIRE,
    "docs": INSTALL_REQUIRES
    + [
        "recommonmark",
        "sphinx==3.1.2",
        "sphinx-markdown-tables",
        "sphinx-rtd-theme==0.4.3",
        "sphinx-copybutton",
    ],
    "client": CLIENT_REQUIRES,
}

setup(
    name="neural-forge-ai",
    description=DOCLINES[0],
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="V Punith Kumar",
    author_email="punithkumar779@gmail.com",
    url="https://github.com/v-punithkumar/Neural-Forge-Ai",
    download_url="https://github.com/v-punithkumar/Neural-Forge-Ai/tags",
    license="Apache 2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    extras_require=EXTRAS_REQUIRE,
    install_requires=INSTALL_REQUIRES,
    entry_points={"console_scripts": ["neuralforge=neural_forge_ai.cli.neural_forge_ai:main"]},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="machine-learning ai neural-forge deep-learning",
    include_package_data=True,
)
