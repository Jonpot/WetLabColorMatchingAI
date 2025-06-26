package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"os"
	"time"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("0 0")
		return
	}
	filePath := os.Args[1]
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		fmt.Printf("0 0")
		return
	}
	var board [][]int
	if err := json.Unmarshal(data, &board); err != nil {
		fmt.Printf("0 0")
		return
	}
	var unknowns [][2]int
	for i, row := range board {
		for j, cell := range row {
			if cell == 0 { // UNKNOWN
				unknowns = append(unknowns, [2]int{i, j})
			}
		}
	}
	if len(unknowns) == 0 {
		fmt.Printf("0 0")
		return
	}
	rand.Seed(time.Now().UnixNano())
	pick := unknowns[rand.Intn(len(unknowns))]
	fmt.Printf("%d %d", pick[0], pick[1])
}
