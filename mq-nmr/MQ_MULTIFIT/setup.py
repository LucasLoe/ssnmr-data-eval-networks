from setuptools import setup, find_packages

setup(
    name='MQ_MULTIFIT',
    version='1.0',
    author='Lucas Löser',
    author_email='lucas_loe@gmx.de',
    description='A package to be used in conjunction with the MiniSpec mq20 and the Baum-Pines MQ-NMR sequence. It provides automatical evaluation of MQ-NMR data.',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'lmfit',
        'pandas',
        'scipy',
        'matplotlib'
    ],
    license='MIT'
)