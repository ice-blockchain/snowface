package snowface

import (
	"fmt"
	"github.com/pkg/errors"
	"math/rand"
	"strings"
)

type SimilarityResp struct {
	UserID    string  `json:"userID"`
	Distance  float64 `json:"distance"`
	BestIndex int8    `json:"bestIndex"`
	Message   string  `json:"message"`
}

func (r *SnowfaceBenchmark) Similarity(worker int) (any, error) {
	var res SimilarityResp
	cl := r.client.R()
	for i := range r.images {
		if i >= 7 {
			break
		}
		cl = cl.SetFileBytes("image", fmt.Sprintf("image%v.jpg", i), r.images[i])
	}
	resp, err := cl.
		EnableForceMultipart().
		SetSuccessResult(&res).
		SetErrorResult(&res).
		SetHeader("Authorization", "Bearer "+r.token[worker]).
		Post(r.host + fmt.Sprintf("/v1w/face-auth/similarity/%[1]v", r.userID[worker]))
	if err != nil || res.Message != "" || resp.StatusCode != 200 {
		if res.Message != "" {
			if strings.Contains(res.Message, "have no registered primary metadata yet") {
				_, err = r.primaryPhoto(worker, r.userID[worker], r.token[worker], func() []byte {
					imgindex := rand.Int31n(int32(len(r.images) - 1))
					return r.images[imgindex]
				})
				if err != nil {
					return nil, err
				}
				return r.Similarity(worker)
			} else {
				err = errors.Errorf("message %v", res.Message)
			}
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
	return &res, nil
}

func (r *SnowfaceBenchmark) SimilarityParse(resp any) {

}
