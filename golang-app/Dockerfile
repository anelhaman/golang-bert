# Use an official Golang image to build the Go application
FROM golang:1.23-alpine AS builder

# Set the Current Working Directory inside the container
WORKDIR /app

# Copy go mod and sum files
COPY go.mod go.sum ./

# Download dependencies
RUN go mod tidy

# Copy the entire project
COPY . .

# Build the Go app
RUN GOOS=linux GOARCH=amd64 go build -o main .

# Start a new stage from scratch for a clean, minimal image
FROM alpine:latest

RUN apk --no-cache add ca-certificates

# Set the Current Working Directory inside the container
WORKDIR /root/

# Copy the pre-built binary from the build container
COPY --from=builder /app/main .

# Expose port 8080 for the Go app
EXPOSE 8080

ENV GIN_MODE=release

# Command to run the Go application
CMD ["./main"]
