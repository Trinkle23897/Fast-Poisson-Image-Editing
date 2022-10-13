SHELL        = /bin/bash
PROJECT_NAME = fpie
PYTHON_FILES = $(shell find setup.py fpie tests docs -type f -name "*.py")
CPP_FILES    = $(shell find fpie -type f -name "*.h" -o -name "*.cc" -o -name "*.cu")
CMAKE_FILES  = $(shell find fpie -type f -name "CMakeLists.txt") $(shell find cmake_modules -type f) CMakeLists.txt
COMMIT_HASH  = $(shell git log -1 --format=%h)

# installation

check_install = python3 -c "import $(1)" || (cd && pip3 install $(1) --upgrade && cd -)
check_install_extra = python3 -c "import $(1)" || (cd && pip3 install $(2) --upgrade && cd -)

flake8-install:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)

py-format-install:
	$(call check_install, isort)
	$(call check_install, yapf)

mypy-install:
	$(call check_install, mypy)

cpplint-install:
	$(call check_install, cpplint)

clang-format-install:
	command -v clang-format-11 || sudo apt-get install -y clang-format-11

cmake-format-install:
	$(call check_install, cmakelang)

doc-install:
	$(call check_install, pydocstyle)
	$(call check_install_extra, doc8, "doc8<1")
	$(call check_install, sphinx)
	$(call check_install, sphinx_rtd_theme)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)

auditwheel-install:
	$(call check_install_extra, auditwheel, auditwheel typed-ast)

# python linter

flake8: flake8-install
	flake8 $(PYTHON_FILES) --count --show-source --statistics --max-line-length 80

py-format: py-format-install
	isort --check $(PYTHON_FILES) && yapf -r -d $(PYTHON_FILES)

mypy: mypy-install
	mypy $(PROJECT_NAME)

# c++ linter

cpplint: cpplint-install
	cpplint $(CPP_FILES)

clang-format: clang-format-install
	clang-format-11 --style=file -i $(CPP_FILES) -n --Werror

cmake-format: cmake-format-install
	cmake-format --check ${CMAKE_FILES}

# documentation

docstyle: doc-install
#	pydocstyle $(PROJECT_NAME)
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

lint: flake8 py-format clang-format cmake-format cpplint mypy docstyle spelling

format: py-format-install clang-format-install cmake-format-install md2rst
	isort $(PYTHON_FILES)
	yapf -ir $(PYTHON_FILES)
	clang-format-11 -style=file -i $(CPP_FILES)
	cmake-format -i ${CMAKE_FILES}

pypi-wheel: auditwheel-install
	ls dist/*.whl | xargs auditwheel repair --plat manylinux_2_17_x86_64
