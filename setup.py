import pathlib
import setuptools

name = "pylibffm"
version = "0.4.0"
author = "ntumlgroup"
license = "MIT License"
license_file = "LICENSE"
description = "A library wrapping libffm"
long_description = (pathlib.Path(__file__).parent / "README.md").read_text()
long_description_content_type = "text/markdown"
url = "https://github.com/Sinacam/pylibffm"
classifiers = [
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
]
packages = setuptools.find_packages()
install_requires = ["scipy", "numpy"]

# Extension modules are completely useless, because they are in effect a dumber and duplicate makefile.
# As it stands, there are __zero__ ways of calling out to make during installation,
# so the current solution is to distribute all binaries and pretend it's part of the source files.
package_data = {"pylibffm": ["wrapper*.so"]}

if __name__ == "__main__":
    setuptools.setup(
        name=name,
        version=version,
        author=author,
        license=license,
        license_file=license_file,
        description=description,
        long_description=long_description,
        long_description_content_type=long_description_content_type,
        url=url,
        # project_urls=project_urls,
        classifiers=classifiers,
        packages=packages,
        package_data=package_data,
        install_requires=install_requires,
    )
