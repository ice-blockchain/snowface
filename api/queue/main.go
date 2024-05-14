package main

import (
	"context"
	appcfg "github.com/ice-blockchain/wintr/config"
	"github.com/ice-blockchain/wintr/log"
	"github.com/ice-blockchain/wintr/server"
	"github.com/imroc/req/v3"
	"github.com/pkg/errors"
	"github.com/rcrowley/go-metrics"
	"sync/atomic"
	"time"
)

const (
	applicationYamlKey = "proxy"
)

type service struct {
	metrics  metrics.Registry
	client   *req.Client
	proxyCfg proxyCfg
	connErrs atomic.Uint64
}

type proxyCfg struct {
	ProxyHostA                   string        `yaml:"proxyHostA"`
	ProxyHostB                   string        `yaml:"proxyHostB"`
	MaxProcessTimeForUnavailable time.Duration `yaml:"maxProcessTimeForUnavailable"`
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	server.New(new(service), applicationYamlKey, "/swagger.html").ListenAndServe(ctx, cancel)
}

func (s *service) RegisterRoutes(router *server.Router) {
	router.Group("/v1w/face-auth").
		POST("primary_photo/:userId", server.RootHandler(s.PrimaryPhoto)).
		POST("liveness/:userId/:sessionId", server.RootHandler(s.Liveness)).
		GET("availability", server.RootHandler(s.Availability))
}

func (s *service) Init(ctx context.Context, cancel context.CancelFunc) {
	appcfg.MustLoadFromKey("proxy", &s.proxyCfg)
	if s.proxyCfg.ProxyHostA == "" || s.proxyCfg.ProxyHostB == "" {
		log.Panic(errors.Errorf("proxyHostA or B not set"))
	}
	s.client = req.DefaultClient()
	s.metrics = metrics.NewRegistry()
	go s.clearHistogram(ctx)
	go metrics.LogScaled(s.metrics, 5*time.Minute, 1*time.Millisecond, s)
}

func (s *service) Close(ctx context.Context) error {
	return nil
}

func (s *service) CheckHealth(ctx context.Context) error {
	return nil
}
