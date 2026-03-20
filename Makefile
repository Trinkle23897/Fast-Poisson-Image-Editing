SHELL        = /bin/bash
PROJECT_NAME = fpie
PYTHON_FILES = $(shell find setup.py fpie tests docs -type f -name "*.py" ! -path "docs/_build/*")
CPP_FILES    = $(shell find fpie -type f -name "*.h" -o -name "*.cc" -o -name "*.cu")
CMAKE_FILES  = $(shell find fpie -type f -name "CMakeLists.txt") $(shell find cmake_modules -type f) CMakeLists.txt
COMMIT_HASH  = $(shell git log -1 --format=%h)
CLANG_FORMAT = $(shell command -v clang-format-11 2>/dev/null || command -v clang-format 2>/dev/null || echo clang-format)

# installation

check_install = python3 -c "import $(1)" || (cd && pip3 install $(1) --upgrade && cd -)
check_install_extra = python3 -c "import $(1)" || (cd && pip3 install $(2) --upgrade && cd -)

ruff-install:
	$(call check_install, ruff)

mypy-install:
	$(call check_install, mypy)

cpplint-install:
	$(call check_install, cpplint)

clang-format-install:
	command -v clang-format-11 || command -v clang-format || (sudo apt-get update && sudo apt-get install -y clang-format)

cmake-format-install:
	$(call check_install, cmakelang)

doc-install:
	$(call check_install_extra, doc8, "doc8<1" setuptools pbr)
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	if command -v apt-get >/dev/null 2>&1; then dpkg -s libenchant-2-dev >/dev/null 2>&1 || (sudo apt-get update && sudo apt-get install -y libenchant-2-dev); fi
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)

auditwheel-install:
	$(call check_install_extra, auditwheel, auditwheel typed-ast)

# python linter

ruff: ruff-install
	ruff check $(PYTHON_FILES)

py-format: ruff-install
	ruff format --check $(PYTHON_FILES)

mypy: mypy-install
	mypy $(PROJECT_NAME)

# c++ linter

cpplint: cpplint-install
	cpplint $(CPP_FILES)

clang-format: clang-format-install
	$(CLANG_FORMAT) --style=file -i $(CPP_FILES) -n --Werror

cmake-format: cmake-format-install
	cmake-format --check ${CMAKE_FILES}

# documentation

docstyle: doc-install
	doc8 docs && cd docs && make html SPHINXOPTS="-W"

doc: doc-install
	cd docs && make html && cd _build/html && python3 -m http.server

spelling: doc-install
	cd docs && make spelling SPHINXOPTS="-W"

doc-clean:
	cd docs && make clean

md2rst:
	pandoc docs/benchmark.md --from markdown --to rst -s -o docs/benchmark.rst
	pandoc docs/report.md --from markdown --to rst -s -o docs/report.rst --columns 100

lint: ruff py-format clang-format cmake-format cpplint mypy docstyle spelling

format: ruff-install clang-format-install cmake-format-install md2rst
	ruff check --fix $(PYTHON_FILES)
	ruff format $(PYTHON_FILES)
	clang-format-11 -style=file -i $(CPP_FILES)
	cmake-format -i ${CMAKE_FILES}

pypi-wheel: auditwheel-install
	ls dist/*.whl | xargs auditwheel repair --plat manylinux_2_17_x86_64
