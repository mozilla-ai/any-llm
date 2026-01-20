#!/bin/bash
set -e

go mod download
go fmt ./...
go vet ./...

# Run golangci-lint if available
if command -v golangci-lint &> /dev/null; then
    golangci-lint run ./...
fi
