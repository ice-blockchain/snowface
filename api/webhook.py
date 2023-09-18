from flask import current_app
import requests, backoff
import logging
from datetime import datetime
def _check_status_code(r):
    return r.response.status_code == 401
def _log(r):
    e = r["exception"]
    logging.error(f"Call to {current_app.config['METADATA_UPDATED_CALLBACK_URL']} failed with {str(e)} {e.response.status_code} {e.response.text}, retrying...")
@backoff.on_exception(backoff.constant,
                      (requests.exceptions.Timeout,
                       requests.exceptions.ConnectionError,
                       requests.HTTPError),
                      giveup=_check_status_code,
                      on_backoff=_log,
                      interval = 0.1,
                      max_tries = 15
                      )
def callback(current_user, primary_md, secondary_md, user):
    time_format = '%Y-%m-%dT%H:%M:%S.%fZ%Z'
    url = current_app.config['METADATA_UPDATED_CALLBACK_URL']
    if not url:
        return
    disabled = False
    if user is not None:
        disabled = user["disabled_at"] is not None and user["disabled_at"] > 0
    lastUpdated = []
    if primary_md is not None:
        lastUpdated = [datetime.utcfromtimestamp(primary_md["uploaded_at"]/1e9).strftime(time_format)]
    if secondary_md is not None:
        lastUpdated.append(datetime.utcfromtimestamp(secondary_md["uploaded_at"]/1e9).strftime(time_format))
    webhook_result = requests.post(url=url, headers={
        "Authorization": f"Bearer {current_user.raw_token}",
        "X-Account-Metadata": current_user.metadata,
        "X-API-Key": current_app.config["METADATA_UPDATED_SECRET"]
    }, json={"lastUpdatedAt":lastUpdated, "disabled": disabled}, verify=False)
    try:
        webhook_result.raise_for_status()
    except requests.HTTPError as e:
        if e.response.status_code == 401:
            raise UnauthorizedFromWebhook(e.response.text)
        else:
            raise e

class UnauthorizedFromWebhook(Exception):
    def __init__(self, message):
        super().__init__(message)