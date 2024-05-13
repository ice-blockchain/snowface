package main

import (
	"context"
	"fmt"
	"github.com/ice-blockchain/wintr/server"
	"github.com/imroc/req/v3"
	"github.com/pkg/errors"
	"io"
	"mime/multipart"
)

type PrimaryPhotoArg struct {
	Image            *multipart.FileHeader `form:"image" formMultipart:"image" swaggerignore:"false"`
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

func (s *service) PrimaryPhoto(
	ctx context.Context,
	request *server.Request[PrimaryPhotoArg, PrimaryPhotoResp],
) (*server.Response[PrimaryPhotoResp], *server.Response[server.ErrorResponse]) {
	var resp PrimaryPhotoResp
	endpoint := fmt.Sprintf("/v1w/face-auth/primary_photo/%s", request.Data.UserID)
	status, errorResp, err := s.callProxy(ctx, s.proxyCfg.ProxyHostB, &proxyCallParams{
		UserID:   request.Data.UserID,
		Endpoint: endpoint,
		Token:    request.Data.Authorization,
		Metadata: request.Data.XAccountMetadata,
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
	}, s.getHistogram(endpoint))
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
