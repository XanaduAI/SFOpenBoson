PYTHON3 := $(shell which python3 2>/dev/null)
COVERAGE3 := $(shell which coverage3 2>/dev/null)

PYTHON := python3
COVERAGE := coverage3
COPTS := run #--append
TESTRUNNER := -m unittest discover sfopenboson/tests

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  install            to install SFOpenBoson"
	@echo "  wheel              to build the SFOpenBoson wheel"
	@echo "  dist               to package the source distribution"
	@echo "  clean              to delete all temporary, cache, and build files"
	@echo "  clean-docs         to delete all built documentation"
	@echo "  test               to run the test suite for all backends"
	@echo "  test-[backend]     to run the test suite for backend fock, tf, or gaussian"
	@echo "  coverage           to generate a coverage report for all backends"
	@echo "  coverage-[backend] to generate a coverage report for backend fock, tf, or gaussian"

.PHONY: install
install:
ifndef PYTHON3
	@echo "To install SFOpenBoson you need to have Python 3 installed"
endif
	$(PYTHON) setup.py install

.PHONY: wheel
wheel:
	$(PYTHON) setup.py bdist_wheel

.PHONY: dist
dist:
	$(PYTHON) setup.py sdist

.PHONY : clean
clean:
	rm -rf sfopenboson/__pycache__
	rm -rf sfopenboson/tests/__pycache__
	rm -rf dist
	rm -rf build

docs:
	make -C doc html

.PHONY : clean-docs
clean-docs:
	make -C doc clean

test:
	$(PYTHON) $(TESTRUNNER)

coverage:
	@echo "Generating coverage report..."
	$(COVERAGE) $(COPTS) $(TESTRUNNER)
	$(COVERAGE) report
	$(COVERAGE) html
