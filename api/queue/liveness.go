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

type LivenessResult struct {
	Result       bool     `json:"result"`
	SessionEnded bool     `json:"sessionEnded"`
	Emotions     []string `json:"emotions"`
	SessionID    string   `json:"sessionId"`
	IONID        string   `json:"IONID"`
}
type LivenessArg struct {
	SessionID        string                  `uri:"sessionId" swaggerignore:"true" required:"true" example:"did:ethr:0x4B73C58370AEfcEf86A6021afCDe5673511376B2"`
	UserID           string                  `uri:"userId" swaggerignore:"true" required:"true" example:"did:ethr:0x4B73C58370AEfcEf86A6021afCDe5673511376B2"`
	Authorization    string                  `header:"Authorization" swaggerignore:"true" required:"true" example:"some token"`
	XAccountMetadata string                  `header:"X-Account-Metadata" swaggerignore:"true" required:"false" example:"some token"`
	Image            []*multipart.FileHeader `form:"image" formMultipart:"image" swaggerignore:"false"`
	Email            string                  `form:"email" swaggerignore:"false" required:"false"`
	PhoneNumber      string                  `form:"phone_number" swaggerignore:"false" required:"false"`
}

func (s *service) Liveness(
	ctx context.Context,
	request *server.Request[LivenessArg, LivenessResult],
) (*server.Response[LivenessResult], *server.Response[server.ErrorResponse]) {
	var resp LivenessResult
	endpoint := fmt.Sprintf("/v1w/face-auth/liveness/%s/%s", request.Data.UserID, request.Data.SessionID)
	status, errorResp, err := s.callProxy(ctx, s.proxyCfg.ProxyHostA, &proxyCallParams{
		UserID:      request.Data.UserID,
		Endpoint:    endpoint,
		Token:       request.Data.Authorization,
		Metadata:    request.Data.XAccountMetadata,
		PhoneNumber: request.Data.PhoneNumber,
		Email:       request.Data.Email,
		payload: func(proxyReq *req.Request) (*req.Request, error) {
			uploads := []req.FileUpload{}
			for _, f := range request.Data.Image {
				uploads = append(uploads, req.FileUpload{
					ParamName: "image",
					FileName:  f.Filename,
					GetFileContent: func() (io.ReadCloser, error) {
						return f.Open()
					},
					FileSize: f.Size,
				})
			}
			return proxyReq.
				SetSuccessResult(&resp).
				SetFileUpload(uploads...), nil
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
	return server.OK[LivenessResult](&resp), nil
}
