from setuptools import setup, find_packages

setup(
  name = 'sincnet-tensorflow',
  packages = find_packages(),
  version = '0.0.1',
  license='MIT',
  description = 'SincNet - Tensorflow',
  author = 'Arya Aftab',
  author_email = 'arya.aftab@gmail.com',
  url = 'https://github.com/AryaAftab/sincnet-tensorflow',
  keywords = [
    'deep learning',
    'tensorflow',
    'sincnet'    
  ],
  install_requires=[
    'numpy>=1.18.5',
    'tensorflow>=2.2'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
