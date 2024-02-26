from setuptools import setup, find_packages

setup(
    name='ml_from_scratch',
    version='0.0.1',
    python_requires='>=3.10',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'scipy'],
    author='João Correia',
    author_email='jfscorreia95@gmail.com',
    description='Machine Learning - Applied Data Science',
    license='MIT',
    keywords='',
)
