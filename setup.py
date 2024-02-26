from setuptools import setup, find_packages

setup(
    name='ml_ads',
    version='0.0.1',
    python_requires='>=3.10',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scipy'],
    author='',
    author_email='',
    description='Machine Learning - Applied Data Science 2024',
    license='MIT',
    keywords='',
)
