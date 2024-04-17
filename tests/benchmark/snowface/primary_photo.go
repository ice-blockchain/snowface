package snowface

import (
	"fmt"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"math/rand"
)

type PrimaryPhotoResp struct {
}

func (r *SnowfaceBenchmark) PrimaryPhoto(worker int) (any, error) {
	// no tokens for random IDs, needed to disable auth on server
	return r.primaryPhoto(worker, uuid.NewString(), "", func() []byte {
		imgindex := rand.Int31n(int32(len(r.images) - 1))
		return r.images[imgindex]
	})
}

type photoPicker func() []byte

func (r *SnowfaceBenchmark) primaryPhoto(worker int, userID, token string, photoPicker photoPicker) (any, error) {
	cl := r.client.R()
	img := photoPicker()
	cl = cl.SetFileBytes("image", fmt.Sprintf("image%v.jpg", 0), img)
	if token != "" {
		cl = cl.SetHeader("Authorization", "Bearer "+token)
	}
	resp, err := cl.
		EnableForceMultipart().
		Post(r.host + fmt.Sprintf("/v1w/face-auth/primary_photo/%[1]v", userID))
	if err != nil || resp.StatusCode != 200 {
		if resp.Response != nil && resp.StatusCode != 200 {
			b, bErr := resp.ToString()
			if bErr != nil {
				panic(bErr)
			}
			err = errors.Errorf("status %v body %v", resp.StatusCode, b)
		}
		return nil, errors.Wrapf(err, "failed to parse face")
	}
	return resp.StatusCode, nil
}

func (r *SnowfaceBenchmark) PrimaryPhotoParse(resp any, b *SnowfaceBenchmark) {

}
