from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'
def get_requirements(file_path: str) -> List[str]:
    """
    This function reads the requirements from a file and returns a list of packages.
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='student_data_prediction',
    version='0.0.1',
    author='Himanshu',
    author_email='himanshurawat4139@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)