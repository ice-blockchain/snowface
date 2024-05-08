package main

import (
	"context"
	"github.com/alitto/pond"
	"github.com/hashicorp/go-multierror"
	appcfg "github.com/ice-blockchain/wintr/config"
	"github.com/ice-blockchain/wintr/connectors/storage/v3"
	"github.com/ice-blockchain/wintr/log"
	"github.com/ice-blockchain/wintr/server"
	"github.com/imroc/req/v3"
	"github.com/minio/minio-go"
	"github.com/pkg/errors"
	"github.com/rcrowley/go-metrics"
	"time"
)

const (
	applicationYamlKey = "proxy"
)

type service struct {
	redis       storage.DB
	minio       *minio.Client
	metrics     metrics.Registry
	client      *req.Client
	proxyCfg    proxyCfg
	dequeuePool *pond.WorkerPool
}
type MinioCfg struct {
	Endpoint        string `yaml:"endpoint"`
	AccessKeyID     string `yaml:"accessKeyID"`
	SecretAccessKey string `yaml:"secretAccessKey"`
	Ssl             bool   `yaml:"ssl"`
}

type proxyCfg struct {
	ProxyHost             string        `yaml:"proxyHost"`
	MaxProcessTimeToQueue time.Duration `yaml:"maxProcessTimeToQueue"`
	DequeueRoutines       int           `yaml:"dequeueRoutines"`
	Minio                 MinioCfg      `yaml:"minio"`
}

func main() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	server.New(new(service), applicationYamlKey, "/swagger.html").ListenAndServe(ctx, cancel)
}

func (s *service) RegisterRoutes(router *server.Router) {
	router.Group("/v1w/face-auth").
		POST("primary_photo/:userId", server.RootHandler(s.PrimaryPhoto)).
		GET("queue/:userId", server.RootHandler(s.PrimaryPhotoQueueStatus)).
		DELETE("queue/:userId", server.RootHandler(s.LeaveQueue))
}

func (s *service) Init(ctx context.Context, cancel context.CancelFunc) {
	var err error
	s.redis = storage.MustConnect(ctx, applicationYamlKey)
	appcfg.MustLoadFromKey("proxy", &s.proxyCfg)
	s.minio, err = minio.New(s.proxyCfg.Minio.Endpoint, s.proxyCfg.Minio.AccessKeyID, s.proxyCfg.Minio.SecretAccessKey, s.proxyCfg.Minio.Ssl)
	log.Panic(err)
	s.client = req.DefaultClient()
	s.metrics = metrics.NewRegistry()
	s.dequeuePool = pond.New(s.proxyCfg.DequeueRoutines, s.proxyCfg.DequeueRoutines)
	go s.clearHistogram(ctx)
	go processQueue[PrimaryPhotoResp](ctx, s)
}

func (s *service) Close(ctx context.Context) error {
	if ctx.Err() != nil {
		return errors.Wrap(ctx.Err(), "could not close repository because context ended")
	}

	return multierror.Append( //nolint:wrapcheck //.
		errors.Wrapf(s.redis.Close(), "could not close redis"),
	).ErrorOrNil()
}

func (s *service) CheckHealth(ctx context.Context) error {
	return nil
}
