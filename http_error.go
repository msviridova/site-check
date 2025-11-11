package main

import "net/http"

type httpError struct {
	Status        int
	ClientMessage string
	LogMessage    string
	RawBody       string
	Err           error
}

func newHTTPError(status int, clientMsg, logMsg string, err error) *httpError {
	return &httpError{
		Status:        status,
		ClientMessage: clientMsg,
		LogMessage:    logMsg,
		Err:           err,
	}
}

func (e *httpError) Error() string {
	if e == nil {
		return ""
	}
	if e.Err != nil {
		return e.Err.Error()
	}
	if e.LogMessage != "" {
		return e.LogMessage
	}
	if e.ClientMessage != "" {
		return e.ClientMessage
	}
	return http.StatusText(e.Status)
}

func (e *httpError) ClientResponse() string {
	if e == nil {
		return http.StatusText(http.StatusInternalServerError)
	}
	if e.ClientMessage != "" {
		return e.ClientMessage
	}
	return http.StatusText(e.Status)
}

func logRouteError(route string, err *httpError) {
	if err == nil {
		return
	}
	fields := map[string]interface{}{"route": route}
	if err.Err != nil {
		fields["error"] = err.Err.Error()
	}
	if err.LogMessage != "" {
		fields["detail"] = err.LogMessage
	}
	logError("handler error", fields)
}
