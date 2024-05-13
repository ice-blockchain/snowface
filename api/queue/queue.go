package main

import (
	"context"
	"github.com/ice-blockchain/wintr/server"
	"github.com/imroc/req/v3"
	"github.com/pkg/errors"
	"github.com/rcrowley/go-metrics"
	"log"
	"net/http"
	"strings"
	"time"
)

const refreshTime = 1 * time.Minute

func (s *service) Availability(
	ctx context.Context,
	request *server.Request[AvailabilityArg, AvailabilityResult],
) (*server.Response[AvailabilityResult], *server.Response[server.ErrorResponse]) {
	primaryPhoto, selfieTime := s.availability("/v1w/face-auth/primary_photo/userId")
	liveness, livenessTime := s.availability("/v1w/face-auth/liveness/userId/sessionId")
	return server.OK[AvailabilityResult](&AvailabilityResult{Available: primaryPhoto && liveness, Liveness: livenessTime.Milliseconds(), PrimaryPhoto: selfieTime.Milliseconds()}), nil
}

type ErrornousResp struct {
	Code    string `json:"code" redis:"code"`
	Message string `json:"message" redis:"message"`
}
type AvailabilityArg struct {
	_ string `swaggerignore:"true" allowUnauthorized:"true"`
}
type AvailabilityResult struct {
	Available    bool  `json:"available"`
	Liveness     int64 `json:"liveness"`
	PrimaryPhoto int64 `json:"primaryPhoto"`
}

type proxyCallParams struct {
	Endpoint string
	UserID   string
	Token    string
	Metadata string
	payload  func(request *req.Request) (*req.Request, error)
}

func (s *service) availability(endpoint string) (bool, time.Duration) {
	histogram := s.getHistogram(endpoint)
	consumedTimeForLastMin := time.Duration(histogram.Percentile(0.95))
	available := consumedTimeForLastMin < s.proxyCfg.MaxProcessTimeForUnavailable
	connErrs := s.connErrs.Load()
	if connErrs > 10 {
		available = false
	}
	return available, consumedTimeForLastMin
}

func (s *service) callProxy(ctx context.Context, host string, params *proxyCallParams, metric metrics.Histogram) (int, *ErrornousResp, error) {
	cl := s.client.R()
	cl = cl.SetHeader("Authorization", params.Token).SetHeader("X-Account-Metadata", params.Metadata)
	var err error
	cl, err = params.payload(cl)
	if err != nil {
		return 0, nil, errors.Wrapf(err, "failed to proxy params")
	}
	var errorResp ErrornousResp
	cl = cl.
		EnableCloseConnection().
		SetContext(ctx).
		EnableForceMultipart().
		SetErrorResult(&errorResp)
	start := time.Now()
	resp, err := cl.Post(host + params.Endpoint)
	if err != nil {
		s.connErrs.Add(1)
		return 0, nil, errors.Wrapf(err, "failed to proxy call to %v", params.Endpoint)
	}
	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusBadRequest {
		metric.Update(int64(time.Since(start)))
	}
	if resp.StatusCode != http.StatusOK {
		return resp.StatusCode, &errorResp, nil
	}
	return resp.StatusCode, nil, nil
}

func (s *service) clearHistogram(ctx context.Context) {
	ticker := time.NewTicker(refreshTime) //nolint:gosec,gomnd // Not an  issue.
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.connErrs.Store(0)
			s.metrics.Each(func(s string, i interface{}) {
				i.(metrics.Histogram).Clear()
			})
		case <-ctx.Done():
			return
		}
	}
}

func (s *service) getHistogram(endpoint string) metrics.Histogram {
	metricKey := s.metricKey(endpoint)
	metric := s.metrics.GetOrRegister(metricKey, metrics.NewHistogram(metrics.NewExpDecaySample(100000, 0.015)))
	return metric.(metrics.Histogram)
}

func (s *service) metricKey(endpoint string) string {
	spl := strings.Split(endpoint, "/")
	lastIdx := 1
	if strings.Contains(endpoint, "/liveness/") {
		lastIdx = 2
	}
	metricKey := strings.Join(spl[0:len(spl)-lastIdx], "/")
	return metricKey
}

func (s *service) Printf(format string, v ...interface{}) {
	log.Printf(format, v...)
}
