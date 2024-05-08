package snowface

import (
	"context"
	"crypto/tls"
	"fmt"
	"github.com/google/uuid"
	"github.com/ice-blockchain/wintr/log"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/hashicorp/go-multierror"
	"github.com/imroc/req/v3"
	"github.com/rcrowley/go-metrics"

	"github.com/ice-blockchain/wintr/auth/fixture"
)

type SnowfaceBenchmark struct {
	host      string
	telemetry metrics.Registry
	routines  int
	images    [][]byte
	session   sync.Map
	userID    map[int]string
	token     map[int]string
	client    *req.Client
	time      time.Duration
}

const localTimer = "responseTime"
const serverElapledTimeFromResp = "serverElapsedTime"
const bytesProcessed = "bytes"
const errs = "errors"
const similarity = "similarity"
const decayAlpha = 0.015
const reservoirSize = 10_000

func NewBenchmark(host string, routines int, t time.Duration, samples string) *SnowfaceBenchmark {
	tel := metrics.NewRegistry()
	err := multierror.Append(
		tel.Register(localTimer, metrics.NewCustomTimer(metrics.NewHistogram(metrics.NewExpDecaySample(reservoirSize, decayAlpha)), metrics.NewMeter())),
		tel.Register(serverElapledTimeFromResp, metrics.NewCustomTimer(metrics.NewHistogram(metrics.NewExpDecaySample(reservoirSize, decayAlpha)), metrics.NewMeter())),
		tel.Register(errs, metrics.NewMeter()),
		tel.Register(similarity, metrics.NewHistogram(metrics.NewExpDecaySample(reservoirSize, decayAlpha))),
	).ErrorOrNil()
	if err != nil {
		panic(err)
	}
	b := &SnowfaceBenchmark{telemetry: tel, routines: routines, time: t, host: host}
	b.loadImages(samples)
	b.client = req.C().SetTLSClientConfig(&tls.Config{InsecureSkipVerify: true}).SetTimeout(10 * time.Minute)
	b.userID = map[int]string{}
	b.token = map[int]string{}
	fmt.Println("Issuing tokens")
	for i := 0; i < routines; i++ {
		uID := uuid.NewString()
		_, token, err := fixture.GenerateIceTokens(uID, "testing-snowface")
		log.Panic(err)
		b.userID[i] = uID
		b.token[i] = token
	}
	fmt.Println("Starting")
	go metrics.LogScaled(tel, time.Minute, time.Millisecond, b)
	return b
}
func (*SnowfaceBenchmark) Printf(format string, v ...interface{}) {
	fmt.Printf(strings.ReplaceAll(format, "timer ", ""), v...)
}

func (r *SnowfaceBenchmark) loadImages(dir string) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		panic(err)
	}
	r.images = make([][]byte, len(entries))
	for i, e := range entries {
		b, err := os.ReadFile(filepath.Join(dir, e.Name()))
		if err != nil {
			panic(err)
		}
		r.images[i] = b
	}
}

type benchmarkFunc func(worker int) (any, error)
type updateFromRespFunc func(resp any, b *SnowfaceBenchmark)

func (r *SnowfaceBenchmark) Benchmark(bench benchmarkFunc, updateFromResp updateFromRespFunc) {
	ctx, cancel := context.WithTimeout(context.Background(), r.time)
	_, err := bench(0)
	if err != nil {
		panic(err)
	}
	for worker := 0; worker < r.routines; worker++ {
		go func(worker int) {
			for ctx.Err() == nil {
				start := time.Now()
				res, bErr := bench(worker)
				if bErr == nil {
					r.telemetry.Get(localTimer).(metrics.Timer).Update(time.Since(start))
				} else {
					r.telemetry.Get(errs).(metrics.Meter).Mark(1)
				}
				if res != nil {
					updateFromResp(res, r)
				}
				if bErr != nil {
					fmt.Println(bErr.Error())
					continue
				}
			}
		}(worker)
	}
	<-ctx.Done()
	cancel()
}
