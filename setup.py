import setuptools
from distutils.command.install import install as DistutilsInstall
import subprocess

name = "pylibffm"
version = "0.1.0"
author = "ntumlgroup"
license = "MIT License"
description = "A library wrapping libffm"

packages = setuptools.find_packages()
install_requires = ["scipy", "numpy"]


class MakeThenInstall(DistutilsInstall):
    def run(self):
        subprocess.call("make", shell=True, executable="/bin/bash")
        DistutilsInstall.run(self)


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
        install_requires=install_requires,
        cmdclass={"install": MakeThenInstall},
    )
