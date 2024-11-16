from setuptools import setup,find_packages
from typing import List

HYPHEN_E_DOT="-e ."

def get_requirements(file_path:str)->List[str]:
    '''This function is responsible for read the packages'''

    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("/n"," ")for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

setup(
    name="home_loan_approval",
    author="sharmi",
    version="0.0.1",
    author_email="anyumsharmila@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)