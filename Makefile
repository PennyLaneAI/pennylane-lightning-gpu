PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3
COVERAGE := --cov=pennylane_lightning_gpu --cov-report term-missing --cov-report=html:coverage_html_report
TESTRUNNER := -m pytest tests --tb=short

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install PennyLane-Lightning-GPU"
	@echo "  wheel              to build the PennyLane-Lightning-GPU wheel"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  test               to run the test suite"
	@echo "  test-cpp           to run the C++ test suite"
	@echo "  test-python        to run the Python test suite"
	@echo "  coverage           to generate a coverage report"
	@echo "  format [check=1]   to apply C++ and Python formatter; use with 'check=1' to check instead of modify (requires black and clang-format)"
	@echo "  format [version=?] to apply C++ and Python formatter; use with 'version={version}' to check or modify with clang-format-{version} instead of clang-format"
	@echo "  check-tidy         to build PennyLane-Lightning-GPU with ENABLE_CLANG_TIDY=ON (requires clang-tidy & CMake)"

.PHONY: install
install:
ifndef CUQUANTUM_SDK
	@echo "Please ensure the CUQUANTUM variable is assigned to the cuQuantum SDK path"
	@test $(CUQUANTUM)
endif
ifndef PYTHON3
	@echo "To install PennyLane-Lightning-GPU you must have Python 3.7+ installed."
endif
	$(PYTHON) setup.py build_ext --cuquantum=$(CUQUANTUM_SDK) --verbose
	$(PYTHON) setup.py install

.PHONY: wheel
wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: dist
dist:
	$(PYTHON) setup.py sdist

.PHONY : clean
clean:
	$(PYTHON) setup.py clean --all
	rm -rf pennylane_lightning_gpu/__pycache__
	rm -rf pennylane_lightning_gpu/src/__pycache__
	rm -rf tests/__pycache__
	rm -rf pennylane_lightning_gpu/src/tests/__pycache__
	rm -rf dist
	rm -rf build
	rm -rf BuildTests Build
	rm -rf .coverage coverage_html_report/
	rm -rf tmp
	rm -rf *.dat
	rm -rf pennylane_lightning_gpu/lightning_gpu_qubit_ops*

docs:
	make -C doc html

.PHONY : clean-docs
clean-docs:
	rm -rf doc/code/api
	make -C doc clean

.PHONY : test-python test-builtin test-suite
test-python: test-builtin test-suite

test-builtin:
	$(PYTHON) -I $(TESTRUNNER)

test-suite:
	pl-device-test --device lightning.gpu --skip-ops --shots=20000
	pl-device-test --device lightning.gpu --shots=None --skip-ops

test-cpp:
	rm -rf ./BuildTests
	cmake . -BBuildTests -DBUILD_TESTS=1
	cmake --build ./BuildTests
	./BuildTests/pennylane_lightning_gpu/src/tests/runner

coverage:
	@echo "Generating coverage report..."
	$(PYTHON) $(TESTRUNNER) $(COVERAGE)
	pl-device-test --device lightning.gpu --skip-ops --shots=20000 $(COVERAGE) --cov-append
	pl-device-test --device lightning.gpu --shots=None --skip-ops $(COVERAGE) --cov-append

.PHONY: format format-cpp format-python
format: format-cpp format-python

format-cpp:
ifdef check
	./bin/format --check --cfversion $(if $(version:-=),$(version),0) pennylane_lightning_gpu/src ./tests
else
	./bin/format --cfversion $(if $(version:-=),$(version),0) pennylane_lightning_gpu/src ./tests
endif

format-python:
ifdef check
	black -l 100 ./pennylane_lightning_gpu ./tests --check
else
	black -l 100 ./pennylane_lightning_gpu ./tests
endif

.PHONY: check-tidy
check-tidy:
	rm -rf ./Build
	cmake . -BBuild -DENABLE_CLANG_TIDY=ON -DBUILD_TESTS=1
	cmake --build ./Build