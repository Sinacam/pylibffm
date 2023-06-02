import setuptools

name = "pylibffm"
version = "0.2.2"
author = "ntumlgroup"
license = "MIT License"
description = "A library wrapping libffm"

packages = setuptools.find_packages()
install_requires = ["scipy", "numpy"]

# Extension modules are completely useless, because they are in effect a dumber and duplicate makefile.
# As it stands, there are __zero__ ways of calling out to make during installation,
# so the current solution is to distribute all binaries and pretend it's part of the source files.
if __name__ == "__main__":
    setuptools.setup(
        name=name,
        version=version,
        author=author,
        license=license,
        # license_file=license_file,
        description=description,
        # long_description=long_description,
        # url=url,
        # project_urls=project_urls,
        # classifiers=classifiers,
        packages=packages,
        package_data={
            "pylibffm": ["wrapper*.so"],
        },
        install_requires=install_requires,
    )
