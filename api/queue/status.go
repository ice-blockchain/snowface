package main

import (
	"context"
	"encoding/json"
	"fmt"
	"github.com/google/uuid"
	"github.com/ice-blockchain/wintr/connectors/storage/v3"
	"github.com/ice-blockchain/wintr/server"
	"github.com/pkg/errors"
	"github.com/redis/go-redis/v9"
	"strings"
)

const (
	emailsKey       = "emails"
	phoneNumbersKey = "phoneNumbers"
)

type StatusArg struct {
	Email       string `form:"email" formMultipart:"email" required:"false" example:"user@example.com"`
	PhoneNumber string `form:"phone_number" formMultipart:"email"  required:"false" example:"+12345678"`
}
type StatusResp struct {
	IONID    string `json:"IONID"`
	Verified bool   `json:"verified"`
}

type User struct {
	PrimaryUploadedAt   uint64 `redis:"primary_uploaded_at"`
	SecondaryUploadedAt uint64 `redis:"secondary_uploaded_at"`
}

func (s *service) Status(
	ctx context.Context,
	request *server.Request[StatusArg, StatusResp],
) (*server.Response[StatusResp], *server.Response[server.ErrorResponse]) {
	userId, ionId, err := s.getId(ctx, request.Data.Email, request.Data.PhoneNumber)
	if err != nil {
		return nil, server.Unexpected(err)
	}
	verified := false
	if ionId == "" {
		ionId, err = generateNewIONId(request.Data.Email, request.Data.PhoneNumber)
		if err != nil {
			return nil, server.BadRequest(err, "MISSING_PROPERTIES")
		}
		return server.OK[StatusResp](&StatusResp{IONID: ionId, Verified: false}), nil
	} else {
		verified, err = s.checkVerified(ctx, userId)
		if err != nil {
			return nil, server.Unexpected(err)
		}
		return server.OK[StatusResp](&StatusResp{IONID: ionId, Verified: verified}), nil
	}
}

func (s *service) getId(ctx context.Context, email string, phoneNumber string) (userId string, ionId string, err error) {
	email = strings.ToLower(strings.TrimSpace(email))
	mappingStr := ""
	if email != "" {
		emailMappingRes := s.redis.HGet(ctx, emailsKey, email)
		if emailMappingRes.Err() != nil && !errors.Is(emailMappingRes.Err(), redis.Nil) {
			return "", "", errors.Wrapf(emailMappingRes.Err(), "failed to get ion id by email %v", email)
		}
		if !errors.Is(emailMappingRes.Err(), redis.Nil) {
			mappingStr = emailMappingRes.Val()
		}
	}
	if phoneNumber != "" {
		phoneMappingRes := s.redis.HGet(ctx, phoneNumbersKey, phoneNumber)
		if phoneMappingRes.Err() != nil && !errors.Is(phoneMappingRes.Err(), redis.Nil) {
			return "", "", errors.Wrapf(phoneMappingRes.Err(), "failed to get ion id by phone %v", phoneNumber)
		}
		if !errors.Is(phoneMappingRes.Err(), redis.Nil) {
			mappingStr = phoneMappingRes.Val()
		}

	}
	if mappingStr == "" {
		return "", "", nil
	}
	var mapping struct {
		UserID string `json:"user_id"`
		IonID  string `json:"ion_id"`
	}
	if err := json.Unmarshal([]byte(mappingStr), &mapping); err != nil {
		return "", "", errors.Wrapf(err, "failed to decode ion_id mapping %v for (%v, %v)", mappingStr, email, phoneNumber)
	}
	return mapping.UserID, mapping.IonID, nil
}

func (s *service) checkVerified(ctx context.Context, userId string) (bool, error) {
	userStates, err := storage.Get[User](ctx, s.redis, userKey(userId))
	if err != nil {
		return false, errors.Wrapf(err, "failed to get user state for user id %v", userId)
	}
	if len(userStates) == 0 {
		return false, nil
	}
	user := userStates[0]
	return user.PrimaryUploadedAt > 0 && user.SecondaryUploadedAt > 0, nil
}

func userKey(userId string) string {
	return fmt.Sprintf("users:%v", userId)
}

func generateNewIONId(email string, phoneNumber string) (string, error) {
	email = strings.ToLower(strings.TrimSpace(email))
	ionId := ""
	if email != "" {
		ionId = uuid.NewSHA1(uuid.MustParse("00000000-0000-0000-0000-000000000000"), []byte(fmt.Sprintf("e:%v", email))).String()
	} else if phoneNumber != "" {
		ionId = uuid.NewSHA1(uuid.MustParse("00000000-0000-0000-0000-000000000000"), []byte(fmt.Sprintf("p:%v", phoneNumber))).String()
	} else {
		return "", errors.New("at least one of (email, phone_number) must be provided")
	}
	return ionId, nil
}
