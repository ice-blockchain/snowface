FROM golang:latest AS build
WORKDIR /app/
COPY . /app/

ENV CGO_ENABLED=0
ENV GOOS=linux
ENV GOARCH=amd64

RUN go build -o bin

FROM gcr.io/distroless/base-debian11:latest
ARG PORT=5002
COPY --from=build /app/bin app
EXPOSE $PORT
ENTRYPOINT ["/app"]