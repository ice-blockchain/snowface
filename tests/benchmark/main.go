package main

import (
	"flag"
	"time"

	"github.com/ice-blockchain/snowface-benchmark/snowface"
)

var folder = flag.String("folder", "", "picks random image from folder and search it")
var routines = flag.Int("routines", 0, "concurrent conns")

var host = flag.String("host", "", "host to connect, localhost by default")

func main() {
	flag.Parse()
	f := ""
	if folder != nil && *folder != "" {
		f = *folder
	}
	if f == "" {
		panic("You must pass --folder")
	}
	r := 4 // 4 workers * 2 threads
	if routines != nil && *routines != 0 {
		r = *routines
	}
	h := "https://localhost:443"
	if host != nil && *host != "" {
		h = *host
	}
	b := snowface.NewBenchmark(h, r, 10*time.Minute, f)
	//b.Benchmark(b.Liveness, b.LivenessParse)
	b.Benchmark(b.PrimaryPhoto, b.PrimaryPhotoParse)
	//b.Benchmark(b.Similarity, b.SimilarityParse)
}
