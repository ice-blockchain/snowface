package snowface

import (
	"fmt"
	"github.com/pkg/errors"
)

type LivenessResp struct {
	Result       bool   `json:"result"`
	SessionEnded bool   `json:"sessionEnded"`
	Message      string `json:"message"`
	SessionID    string `json:"sessionId"`
}

func (r *SnowfaceBenchmark) Liveness(worker int) (any, error) {
	var res LivenessResp
	cl := r.client.R()
	for i := range r.images {
		cl = cl.SetFileBytes("image", fmt.Sprintf("image%v.jpg", i), r.images[i])
	}
	session, ok := r.session.Load(worker)
	if !ok {
		if err := r.Session(worker); err != nil {
			panic(err)
		}
		session, ok = r.session.Load(worker)
	}
	resp, err := cl.
		EnableForceMultipart().
		SetSuccessResult(&res).
		SetErrorResult(&res).
		SetHeader("Authorization", "Bearer "+r.token[worker]).
		Post(r.host + fmt.Sprintf("/v1w/face-auth/liveness/%[1]v/%[2]v", r.userID[worker], session.(string)))
	if err != nil || res.Message != "" || resp.StatusCode != 200 {
		if res.Message != "" {
			err = errors.Errorf("message %v", res.Message)
		}
		if resp.Response != nil && resp.StatusCode != 200 {
			b, bErr := resp.ToString()
			if bErr != nil {
				panic(bErr)
			}
			err = errors.Errorf("status %v body %v", resp.StatusCode, b)
		}
		return nil, errors.Wrapf(err, "failed to parse face")
	}
	if res.SessionID != session {
		r.session.Store(worker, res.SessionID)
	}
	if res.SessionEnded {
		if err := r.Session(worker); err != nil {
			return nil, errors.Wrapf(err, "failed fetch new session")
		}
	}
	return &res, nil
}

func (r *SnowfaceBenchmark) LivenessParse(resp any) {

}

func (r *SnowfaceBenchmark) Session(worker int) error {
	type response struct {
		SessionID string `json:"sessionId"`
		Message   string `json:"message"`
	}
	var res response
	resp, err := r.client.R().
		SetSuccessResult(&res).
		SetHeader("Authorization", "Bearer "+r.token[worker]).
		Post(r.host + fmt.Sprintf("/v1w/face-auth/emotions/%[1]v", r.userID[worker]))
	if err != nil || res.Message != "" || resp.StatusCode != 200 {
		if res.Message != "" {
			err = errors.Errorf("message %v", res.Message)
		}
		if resp.Response != nil && resp.StatusCode != 200 {
			b, bErr := resp.ToString()
			if bErr != nil {
				panic(bErr)
			}
			err = errors.Errorf("status %v body %v", resp.StatusCode, b)
		}
		return errors.Wrapf(err, "failed to parse face")
	}
	r.session.Store(worker, res.SessionID)
	return nil
}
