from setuptools import setup, find_packages
setup(
    name="ml",
    version="0",
    packages=find_packages("src"),
    package_dir={"":"src"},

    install_requires=[  
        'beautifulsoup4==4.9.1',
        'gym==0.17.2',
        'jupyter==1.0.0',
        'matplotlib==3.2.1',
        'nltk==3.5',
        'numpy==1.18.4',
        'requests==2.31.0',
        'scikit-learn==0.23.1',
    ],
    extras_require={
        "testing": ["pytest"],
    },
    entry_points={
        'console_scripts': [
            'nnt = ml.pytorch.main:main',
        ]
    }
)
