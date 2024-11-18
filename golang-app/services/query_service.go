package services

import (
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/go-resty/resty/v2"
)

// sendToBERT sends text to the BERT API and retrieves the response
func SendToBERT(text string) (string, error) {
	// Get BERT API URL from environment variable
	url := os.Getenv("BERT_API_URL")
	if url == "" {
		url = "http://localhost:5000/bert" // Default URL if not set
	}

	// Create a Resty client
	client := resty.New()

	// Send POST request
	resp, err := client.R().
		SetHeader("Content-Type", "application/json").
		SetBody(map[string]string{"text": text}).
		Post(url)

	if err != nil {
		return "", fmt.Errorf("failed to send request to BERT API: %w", err)
	}

	// Check for errors in the response
	if resp.IsError() {
		return "", fmt.Errorf("BERT API error: %s", resp.String())
	}

	// Parse the JSON response
	var responseMap map[string]string
	err = json.Unmarshal(resp.Body(), &responseMap)
	if err != nil {
		return "", fmt.Errorf("failed to parse JSON response: %w", err)
	}

	// Extract the "answer" field
	answer, ok := responseMap["answer"]
	if !ok {
		return "", fmt.Errorf("response does not contain 'answer' field")
	}

	log.Printf("\n ++++++++ %v\n===== %v\n", text, answer)

	// Return the extracted answer
	return answer, nil
}
