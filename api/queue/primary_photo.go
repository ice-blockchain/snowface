package main

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/ice-blockchain/wintr/server"
	"github.com/imroc/req/v3"
	"github.com/pkg/errors"
	"io"
	"mime/multipart"
)

type PrimaryPhotoArg struct {
	Image            *multipart.FileHeader `form:"image" formMultipart:"image" swaggerignore:"true"`
	UserID           string                `uri:"userId" swaggerignore:"true" required:"true" example:"did:ethr:0x4B73C58370AEfcEf86A6021afCDe5673511376B2"`
	Authorization    string                `header:"Authorization" swaggerignore:"true" required:"true" example:"some token"`
	XAccountMetadata string                `header:"X-Account-Metadata" swaggerignore:"true" required:"false" example:"some token"`
}
type UserID struct {
	UserID string `uri:"userId" swaggerignore:"true" required:"true" example:"did:ethr:0x4B73C58370AEfcEf86A6021afCDe5673511376B2"`
}
type PrimaryPhotoResp struct {
	SkipEmotions bool `json:"skipEmotions" redis:"skipEmotions"`
}

func (s *service) PrimaryPhotoQueueStatus(
	ctx context.Context,
	request *server.Request[UserID, PrimaryPhotoResp],
) (*server.Response[PrimaryPhotoResp], *server.Response[server.ErrorResponse]) {
	fmt.Println(request.Data.UserID)
	pending := s.redis.SIsMember(ctx, redisSelfieQueue, request.Data.UserID)
	if pending.Err() != nil {
		return nil, server.Unexpected(pending.Err())
	}
	if pending.Val() {
		return nil, &server.Response[server.ErrorResponse]{
			Data: (&server.ErrorResponse{
				Code:  "QUEUED",
				Error: "still pending in queue",
			}).Fail(errors.Errorf("still pending in queue")),
			Headers: nil,
			Code:    429,
		}
	}
	d := s.redis.Get(ctx, fmt.Sprintf("queue:%v", request.Data.UserID))
	if d.Err() != nil {
		return nil, server.Unexpected(d.Err())
	}
	var resp queueResp[PrimaryPhotoResp]
	if err := json.Unmarshal([]byte(d.Val()), &resp); err != nil {
		return nil, server.Unexpected(err)
	}
	if resp.Err != nil {
		return nil, &server.Response[server.ErrorResponse]{
			Code: resp.Status,
			Data: (&server.ErrorResponse{
				Error: resp.Err.Message,
				Code:  resp.Err.Code,
			}).Fail(errors.Errorf(resp.Err.Message)),
		}
	}
	return server.OK(&resp.Success), nil
}

func (s *service) PrimaryPhoto(
	ctx context.Context,
	request *server.Request[PrimaryPhotoArg, PrimaryPhotoResp],
) (*server.Response[PrimaryPhotoResp], *server.Response[server.ErrorResponse]) {
	var resp PrimaryPhotoResp
	status, errorResp, err := s.callProxyOrQueue(ctx, &proxyCallParams{
		UserID:   request.Data.UserID,
		Endpoint: fmt.Sprintf("/v1w/face-auth/primary_photo/%s", request.Data.UserID),
		Token:    request.Data.Authorization,
		Metadata: request.Data.XAccountMetadata,
		FileSize: request.Data.Image.Size,
		file: func() (io.ReadCloser, error) {
			return request.Data.Image.Open()
		},
		payload: func(proxyReq *req.Request) (*req.Request, error) {
			return proxyReq.
				SetSuccessResult(&resp).
				SetFileUpload(req.FileUpload{
					ParamName: "image",
					FileName:  request.Data.Image.Filename,
					GetFileContent: func() (io.ReadCloser, error) {
						return request.Data.Image.Open()
					},
					FileSize: request.Data.Image.Size,
				}), nil
		},
	})
	if err != nil {
		return nil, server.Unexpected(err)
	}
	if errorResp != nil {
		return nil, &server.Response[server.ErrorResponse]{
			Code: status,
			Data: (&server.ErrorResponse{
				Error: errorResp.Message,
				Code:  errorResp.Code,
			}).Fail(errors.Errorf(errorResp.Message)),
		}
	}
	return server.OK[PrimaryPhotoResp](&PrimaryPhotoResp{SkipEmotions: resp.SkipEmotions}), nil
}
