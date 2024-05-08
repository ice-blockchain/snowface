package main

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/ice-blockchain/wintr/log"
	"github.com/ice-blockchain/wintr/server"
	"github.com/imroc/req/v3"
	"github.com/minio/minio-go"
	"github.com/pkg/errors"
	"github.com/rcrowley/go-metrics"
	"github.com/redis/go-redis/v9"
	"io"
	"net/http"
	"strings"
	"time"
)

const bucketSelfieQueue = "queue"
const redisSelfieQueue = "queue"

type ErrornousResp struct {
	Code    string `json:"code" redis:"code"`
	Message string `json:"message" redis:"message"`
}

type proxyCallParams struct {
	Endpoint string `json:"endpoint"`
	UserID   string `json:"userID"`
	file     req.GetContentFunc
	FileSize int64  `json:"fileSize"`
	Token    string `json:"token"`
	Metadata string `json:"metadata"`
	payload  func(request *req.Request) (*req.Request, error)
}

type queueResp[T any] struct {
	Err       *ErrornousResp `json:"err"`
	UserID    string         `json:"userID"`
	Success   T              `json:"success"`
	Status    int            `json:"status"`
	Completed bool           `json:"completed"`
}

func (s *service) callProxyOrQueue(ctx context.Context, params *proxyCallParams) (int, *ErrornousResp, error) {
	spl := strings.Split(params.Endpoint, "/")
	metricKey := strings.Join(spl[0:len(spl)-2], "/")
	metric := s.metrics.GetOrRegister(metricKey, metrics.NewHistogram(metrics.NewExpDecaySample(100000, 0.015)))
	consumedTimeForLastMin := time.Duration(metric.(metrics.Histogram).Percentile(0.95))
	enqueue := metric.(metrics.Histogram).Count() > 30 && consumedTimeForLastMin > s.proxyCfg.MaxProcessTimeToQueue
	userAlreadyPassedQueue := s.redis.Get(ctx, fmt.Sprintf("queue:%v", params.UserID))
	if enqueue && userAlreadyPassedQueue.Val() != "" {
		enqueue = false
	}
	if enqueue {
		return s.enqueue(ctx, params, metric.(metrics.Histogram))
	} else {
		return s.callProxy(ctx, params, metric.(metrics.Histogram))
	}
}

func processQueue[T any](ctx context.Context, s *service) {
	if s.proxyCfg.DequeueRoutines > 0 {
		ticker := time.NewTicker(time.Minute) //nolint:gosec,gomnd // Not an  issue.
		defer ticker.Stop()

		for {
			queuedUserID := s.redis.SPop(ctx, redisSelfieQueue)
			if queuedUserID.Err() != nil {
				select {
				case <-ticker.C:
					continue
				case <-ctx.Done():
					return
				}
			}
			var param proxyCallParams
			data := s.redis.Get(ctx, fmt.Sprintf("queue:%v", queuedUserID.Val()))
			if data.Err() != nil {
				continue
			}
			err := json.Unmarshal([]byte(data.Val()), &param)
			if err != nil {
				log.Error(errors.Wrapf(err, "failed to dequeue request %v", param.UserID))
				continue
			}
			s.dequeuePool.Submit(func() {
				var resp T
				spl := strings.Split(param.Endpoint, "/")
				metricKey := strings.Join(spl[0:len(spl)-2], "/")
				status, errResp, err := s.callProxy(ctx, &proxyCallParams{
					Endpoint: param.Endpoint,
					UserID:   param.UserID,
					file: func() (io.ReadCloser, error) {
						return s.minio.GetObjectWithContext(ctx, bucketSelfieQueue, "selfie-"+param.UserID, minio.GetObjectOptions{})
					},
					Token:    param.Token,
					Metadata: param.Metadata,
					payload: func(proxyReq *req.Request) (*req.Request, error) {
						return proxyReq.
							SetSuccessResult(&resp).
							SetFileUpload(req.FileUpload{
								ParamName: "image",
								FileName:  fmt.Sprintf("image0.jpg"),
								GetFileContent: func() (io.ReadCloser, error) {
									return s.minio.GetObjectWithContext(ctx, bucketSelfieQueue, "selfie-"+param.UserID, minio.GetObjectOptions{})
								},
								FileSize: param.FileSize,
							}), nil
					},
				}, s.metrics.GetOrRegister(metricKey, metrics.NewHistogram(metrics.NewExpDecaySample(100000, 0.015))).(metrics.Histogram))
				if err != nil {
					log.Error(errors.Wrapf(err, "failed to process request from queue %v", param.UserID))
				}
				enc, _ := json.Marshal(queueResp[T]{
					Err:       errResp,
					UserID:    param.UserID,
					Success:   resp,
					Status:    status,
					Completed: true,
				})
				if sErr := s.redis.Set(ctx, fmt.Sprintf("queue:%v", param.UserID), string(enc), time.Duration(0)); sErr.Err() != nil {
					log.Error(errors.Wrapf(err, "failed to store result from queue %v", param.UserID))
				}
				if err = s.minio.RemoveObject(bucketSelfieQueue, "selfie-"+param.UserID); err != nil {
					log.Error(errors.Wrapf(err, "failed to remove tmp pic from queue %v", param.UserID))
				}
			})
		}
	}
}

func (s *service) enqueue(ctx context.Context, params *proxyCallParams, metric metrics.Histogram) (int, *ErrornousResp, error) {
	queueItem, err := json.Marshal(params)
	if err != nil {
		return 0, nil, errors.Wrapf(err, "failed to encode enqueueing item %v", params)
	}
	res := s.redis.SAdd(ctx, redisSelfieQueue, params.UserID)
	if res.Err() != nil {
		return 0, nil, errors.Wrapf(err, "failed to enqueue user %v", params.UserID)
	}
	if res.Val() == 0 {
		return 429, &ErrornousResp{
			Message: "operation is enqueued",
			Code:    "QUEUED",
		}, nil
	}
	if err = s.redis.Set(ctx, fmt.Sprintf("queue:%v", params.UserID), queueItem, time.Duration(0)).Err(); err != nil {
		return 0, nil, errors.Wrapf(err, "failed to enqueue user %v", string(queueItem))
	}
	file, _ := params.file()
	if _, err := s.minio.PutObjectWithContext(ctx, bucketSelfieQueue, "selfie-"+params.UserID, file, params.FileSize, minio.PutObjectOptions{}); err != nil {
		return 0, nil, errors.Wrapf(err, "failed to store user selfie to minio queue")
	}
	return 429, &ErrornousResp{
		Message: fmt.Sprintf("operation is enqueued (%v/%v)", time.Duration(metric.Percentile(0.95)), s.proxyCfg.MaxProcessTimeToQueue),
		Code:    "QUEUED",
	}, nil
}

func (s *service) callProxy(ctx context.Context, params *proxyCallParams, metric metrics.Histogram) (int, *ErrornousResp, error) {
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
	resp, err := cl.Post(s.proxyCfg.ProxyHost + params.Endpoint)
	if err != nil {
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
	ticker := time.NewTicker(time.Minute) //nolint:gosec,gomnd // Not an  issue.
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			s.metrics.Each(func(s string, i interface{}) {
				i.(metrics.Histogram).Clear()
			})
		case <-ctx.Done():
			return
		}
	}
}

func (s *service) LeaveQueue(
	ctx context.Context,
	request *server.Request[UserID, struct{}],
) (*server.Response[struct{}], *server.Response[server.ErrorResponse]) {
	d := s.redis.SRem(ctx, redisSelfieQueue, request.Data.UserID)
	if d.Err() != nil && !errors.Is(d.Err(), redis.Nil) {
		return nil, server.Unexpected(d.Err())
	}

	return server.OK(&struct{}{}), nil
}
