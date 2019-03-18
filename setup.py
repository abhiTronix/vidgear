import setuptools
from setuptools import setup

setup(
    name='vidgear',
    packages=['vidgear','vidgear.gears'],
    version='0.1.0',
    description='A OpenCV Python Multi-Threaded Video Streaming Wrapper Library',
    license='MIT License',
    author='abhiTronix',
    author_email='abhi.una12@gmail.com',
    url='https://github.com/abhiTronix/vidgear',
    download_url='https://github.com/abhiTronix/vidgear/tarball/0.1.0',
    keywords=['computer vision', 'multi-thread', 'opencv', 'opencv2', 'opencv4', 'picamera'],
    classifiers=[
    'Development Status :: 5 - Production/Stable',
    'Intended Audience :: Developers',
    'Topic :: Computer Vision :: Video Processing',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',],
    python_requires='>=2.7',
    scripts=[],
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/abhiTronix/vidgear/issues',
        'Funding': 'https://paypal.me/AbhiTronix?locale.x=en_GB',
        'Source': 'https://github.com/abhiTronix/vidgear',},
)
