from setuptools import setup, find_packages
setup(
    name="nntraining",
    version="0",
    packages=find_packages("src"),
    package_dir={"":"src"},

    install_requires=[  
        'beautifulsoup4==4.7.1',
        'matplotlib==3.0.3',
        'requests==2.21.0',
        'nltk==3.4',
        'numpy==1.16.1',
        'torch==1.0.1',
    ],
    extras_require={
        "testing": ["pytest"],
    },
    entry_points={
        'console_scripts': [
            'nnt = nntraining.pytorch.main:main',
        ]
    }
)
