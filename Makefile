.DEFAULT_GOAL := help
SHELL := /bin/bash

IMAGE ?= oratio-api:local
PLATFORM ?= linux/arm64
MAX_IMAGE_MB ?= 2500
AWS_PROFILE ?= oratio
STACK ?= oratio-api
SPIKE ?= 0
REGION ?= ap-south-1
MEMORY ?= 3008
AWS_ARGS = --profile $(AWS_PROFILE) --region $(REGION)
ECR = $(shell aws sts get-caller-identity $(AWS_ARGS) --query Account --output text).dkr.ecr.$(REGION).amazonaws.com

.PHONY: help
help: ## List targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

.PHONY: sync
sync: ## Install the locked dependencies into the active environment
	VIRTUAL_ENV=$$CONDA_PREFIX uv sync --frozen --active --inexact

.PHONY: lint
lint: ## Ruff lint and format check
	ruff check .
	ruff format --check .

.PHONY: typecheck
typecheck: ## mypy --strict on the library and the API
	mypy packages/speechgrade/src services/api/src

.PHONY: test
test: ## Unit tests with coverage
	pytest --cov --cov-report=term-missing --cov-fail-under=85

.PHONY: verify
verify: lint typecheck test ## Everything CI runs for Python

.PHONY: dev
dev: ## Run the API locally with reload
	ORATIO_ENABLE_SPIKE=1 uvicorn oratio_api.main:app --reload --port 8000

.PHONY: image
image: ## Build the Lambda container image
	docker build --platform $(PLATFORM) -f services/api/Dockerfile -t $(IMAGE) .

.PHONY: image-size
image-size: ## Fail if the image exceeds MAX_IMAGE_MB uncompressed
	@size=$$(( $$(docker image inspect $(IMAGE) --format '{{.Size}}') / 1000000 )); \
	echo "$(IMAGE): $$size MB (limit $(MAX_IMAGE_MB) MB)"; \
	test $$size -le $(MAX_IMAGE_MB)

.PHONY: image-run
image-run: ## Run the built image locally on port 8000
	docker run --rm -p 8000:8000 -e ORATIO_ENABLE_SPIKE=1 $(IMAGE)

.PHONY: deploy
deploy: ## Push the arm64 image to ECR and deploy the API stack
	aws ecr describe-repositories --repository-names $(STACK) $(AWS_ARGS) >/dev/null 2>&1 \
		|| aws ecr create-repository --repository-name $(STACK) $(AWS_ARGS) >/dev/null
	aws ecr get-login-password $(AWS_ARGS) | docker login --username AWS --password-stdin $(ECR)
	docker buildx build --platform linux/arm64 --provenance=false --push \
		-f services/api/Dockerfile -t $(ECR)/$(STACK):latest .
	aws cloudformation deploy --template-file infra/template.yaml --stack-name $(STACK) $(AWS_ARGS) \
		--capabilities CAPABILITY_IAM CAPABILITY_AUTO_EXPAND --no-fail-on-empty-changeset \
		--parameter-overrides MemorySize=$(MEMORY) EnableSpike=$(SPIKE) \
			ImageUri=$(ECR)/$(STACK)@$$(docker buildx imagetools inspect $(ECR)/$(STACK):latest --format '{{.Manifest.Digest}}')
	aws cloudformation describe-stacks --stack-name $(STACK) $(AWS_ARGS) \
		--query 'Stacks[0].Outputs[?OutputKey==`ApiUrl`].OutputValue' --output text

.PHONY: destroy
destroy: ## Remove the API stack and its ECR repository
	aws cloudformation delete-stack --stack-name $(STACK) $(AWS_ARGS)
	aws cloudformation wait stack-delete-complete --stack-name $(STACK) $(AWS_ARGS)
	aws ecr delete-repository --repository-name $(STACK) --force $(AWS_ARGS) >/dev/null

.PHONY: budget
budget: ## Create the $$10 / $$25 cost alarms (EMAIL=you@example.com)
	@test -n "$(EMAIL)" || { echo "usage: make budget EMAIL=you@example.com"; exit 1; }
	sed 's/BUDGET_EMAIL/$(EMAIL)/g' infra/budgets.json > $(TMPDIR)/oratio-budget.json
	aws budgets create-budget --profile $(AWS_PROFILE) \
		--account-id $$(aws sts get-caller-identity --profile $(AWS_PROFILE) --query Account --output text) \
		--cli-input-json file://$(TMPDIR)/oratio-budget.json
	rm -f $(TMPDIR)/oratio-budget.json
