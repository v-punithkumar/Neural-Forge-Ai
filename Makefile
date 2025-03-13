.PHONY: quality style test

# Check that source code meets quality standards
quality:
	black --check --line-length 119 --target-version py38 .
	isort --check-only .
	flake8 --max-line-length 119

# Format source code automatically
style:
	black --line-length 119 --target-version py38 .
	isort .

# Run tests
test:
	pytest -sv ./src/

# Build and push Docker image to Docker Hub
docker:
	docker build -t neural-forge-ai:latest .
	docker tag neural-forge-ai:latest punithkumar779/neural-forge-ai:latest
	docker push punithkumar779/neural-forge-ai:latest

# Build and push API Docker image to Docker Hub
api:
	docker build -t neural-forge-ai-api:latest -f Dockerfile.api .
	docker tag neural-forge-ai-api:latest punithkumar779/neural-forge-ai-api:latest
	docker push punithkumar779/neural-forge-ai-api:latest

# NVIDIA NGC Container Registry (if applicable)
ngc:
	docker build -t neural-forge-ai:latest .
	docker tag neural-forge-ai:latest nvcr.io/ycymhzotssoi/neural-forge-ai:latest
	docker push nvcr.io/ycymhzotssoi/neural-forge-ai:latest

# Build and upload Python package
pip:
	rm -rf build/
	rm -rf dist/
	make style && make quality
	python setup.py sdist bdist_wheel
	twine upload dist/* --verbose --repository neural-forge-ai
