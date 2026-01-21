#!/bin/bash
set -e

go mod download
go test -v ./...
