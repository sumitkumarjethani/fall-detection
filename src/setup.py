from setuptools import setup, find_packages

requirements = []
with open("requirements.txt") as f:
    requirements = f.readlines()

setup(
    name="fall_detection",
    version="0.1",
    description="Fall Detection",
    author="Vito Stamatti, Sumit Kumar Jethani Jethani",
    license="MIT",
    packages=find_packages(),
    # install_requires=requirements,
)
