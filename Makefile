.PHONY: io phase0 baseline settling ordering predictive adaptive pat test

io:
	python -m dcopt.io --config config.yaml

phase0:
	python -m dcopt.phase0_data_foundation --config config.yaml

baseline:
	python -m dcopt.baseline --config config.yaml

settling:
	python -m dcopt.settling --config config.yaml

ordering:
	python -m dcopt.ordering --config config.yaml

predictive:
	python -m dcopt.predictive --config config.yaml

adaptive:
	python -m dcopt.adaptive --config config.yaml

pat:
	python -m dcopt.pat --config config.yaml

test:
	python -m pytest
