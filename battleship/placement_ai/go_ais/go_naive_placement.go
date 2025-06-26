package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"sort"
)

type shipSpec struct {
	Length int `json:"length"`
	Count  int `json:"count"`
}

type config struct {
	BoardShape []int               `json:"board_shape"` // [rows, cols]
	ShipSchema map[string]shipSpec `json:"ship_schema"`
}

type placement struct {
	Row       int    `json:"row"`
	Col       int    `json:"col"`
	Length    int    `json:"length"`
	Direction string `json:"direction"` // always "horizontal"
}

func main() {
	if len(os.Args) < 2 {
		fmt.Print("[]")
		return
	}

	// ---------- read & decode ----------
	data, err := ioutil.ReadFile(os.Args[1])
	if err != nil {
		fmt.Print("[]")
		return
	}

	var cfg config
	if err := json.Unmarshal(data, &cfg); err != nil {
		fmt.Print("[]")
		return
	}
	if len(cfg.BoardShape) != 2 {
		fmt.Print("[]")
		return
	}
	rows, cols := cfg.BoardShape[0], cfg.BoardShape[1]

	// ---------- generate placements ----------
	var placements []placement
	curRow, curCol := 0, 0

	// iterate deterministically by sorting ship names
	names := make([]string, 0, len(cfg.ShipSchema))
	for name := range cfg.ShipSchema {
		names = append(names, name)
	}
	sort.Strings(names)

	for _, name := range names {
		spec := cfg.ShipSchema[name]
		for i := 0; i < spec.Count; i++ {
			if curCol+spec.Length > cols { // wrap to next row
				curRow++
				curCol = 0
				if curRow >= rows { // board overflow: just stop
					goto done
				}
			}
			placements = append(placements, placement{
				Row:       curRow,
				Col:       curCol,
				Length:    spec.Length,
				Direction: "horizontal",
			})
			curCol += spec.Length + 1 // leave 1-cell gap
		}
	}
done:

	// ---------- output ----------
	out, err := json.Marshal(placements)
	if err != nil {
		fmt.Print("[]")
		return
	}
	fmt.Print(string(out))
}
