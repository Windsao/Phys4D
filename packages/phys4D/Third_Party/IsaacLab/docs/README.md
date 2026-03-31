

We use [Sphinx](https:

> **Note:** To avoid dependency conflicts, we strongly recommend using a Python virtual environment to isolate the required dependencies from your system's global Python environment.



This section describes how to build the documentation for the current version of the project.

<details open>
<summary><strong>Linux</strong></summary>

```bash

cd docs
pip install -r requirements.txt


make current-docs


xdg-open _build/current/index.html
```
</details>

<details> <summary><strong>Windows</strong></summary>

```batch
:: 1. Navigate to the docs directory and install dependencies
cd docs
pip install -r requirements.txt

:: 2. Build the current documentation
make current-docs

:: 3. Open the current docs
start _build\current\index.html
```
</details>




This section describes how to build the multi-version documentation, which includes previous tags and the main branch.

<details open> <summary><strong>Linux</strong></summary>

```bash

cd docs
pip install -r requirements.txt


make multi-docs


xdg-open _build/index.html
```
</details>

<details> <summary><strong>Windows</strong></summary>

```batch
:: 1. Navigate to the docs directory and install dependencies
cd docs
pip install -r requirements.txt

:: 2. Build the multi-version documentation
make multi-docs

:: 3. Open the multi-version docs
start _build\index.html
```
</details>
