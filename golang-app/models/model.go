package models

// PredictionResponse represents the structure of the BERT API response
type PredictionResponse struct {
	Predictions [][]float64 `json:"predictions"`
}
