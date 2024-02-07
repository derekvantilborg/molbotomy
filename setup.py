from setuptools import setup

setup(
    name='molbotomy',
    version='0.0.1',
    packages=['molbotomy',
              'molbotomy.cleaning',
              'molbotomy.constants',
              'molbotomy.descriptors',
              'molbotomy.distances',
              'molbotomy.evaluate',
              'molbotomy.split',
              'molbotomy.tools',
              'molbotomy.utils'],
    url='https://github.com/molML/MolBotomy',
    license='MIT',
    author='Derek van Tilborg',
    author_email='',
    description='Tool to clean, process, and split molecular data',
    install_requires=['tqdm', 'scipy', 'numpy', 'rdkit', 'levenshtein']
)
