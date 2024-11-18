package main

import (
	"golang-app/handlers"

	"github.com/gin-gonic/gin"
)

func main() {
	// Create a new Gin router
	router := gin.Default()

	// Define an endpoint for querying BERT
	router.POST("/answer", handlers.HandleQuery)

	// Start the server on port 8080
	router.Run(":8080")
}
