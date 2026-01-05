from mypyc.doc.conf import author
from setuptools import setup, find_packages
HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->list[str]:
    '''this function will return the list of requirements from a requirements file'''
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements= [req.replace('\n','') for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements



setup(
    name='end_to_end_ML_project',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=get_requirements('requirements.txt'),
    dependency_links=[],
   author='chinthaka2000',
   author_email='smartchinthaka512@gmail.com',
)