package handlers

import (
	"net/http"

	"golang-app/services"

	"github.com/gin-gonic/gin"
)

// handleQuery is the Gin handler for processing client queries
func HandleQuery(c *gin.Context) {
	// Parse the query parameter from the request
	var requestBody struct {
		Text string `json:"text"`
	}
	if err := c.ShouldBindJSON(&requestBody); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request body"})
		return
	}

	// Call sendToBERT to process the text
	bertResponse, err := services.SendToBERT(requestBody.Text)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	// Return the response to the client
	c.JSON(http.StatusOK, gin.H{"response": bertResponse})
}
