How to setup the poetry environment
===

1. install python 3 (3.10.8)
2. Install pipx ``pip install pipx``
3. Install poetry ``pipx install poetry``
4. Set python 3.10 as active environment ``poetry env use python3.10``
5. (Optional) Confirm that the correct version is activated with ``poetry env list``
6. Install dependencies: Run ``poetry install`` (may take a while since torch has a lot of stuff)

How to add a package
---
* Run ``poetry add <package>``