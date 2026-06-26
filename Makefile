.PHONY: test serve smoke

PYTHON ?= python3
PORT ?= 7860

test:
	$(PYTHON) -m unittest discover -s tests
	$(PYTHON) -m py_compile web_interface.py main.py am/*.py arc3/*.py

serve:
	$(PYTHON) web_interface.py --port $(PORT)

smoke:
	$(PYTHON) main.py --test
