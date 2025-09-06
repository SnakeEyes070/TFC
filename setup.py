from setuptools import setup, find_packages

setup(
    name="traffic-violation-app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "flask==2.3.3",
        "opencv-python-headless==4.8.1.78",
        "numpy==1.24.3",
        "gunicorn==21.2.0",
        "pillow==10.0.1"
    ],
)