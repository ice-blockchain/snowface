from flask import Blueprint, request, current_app, Response
import service, exceptions, webhook

from auth import auth_required, Token
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from limits import parse
from webhook import UnauthorizedFromWebhook
import logging

from prometheus_client import CONTENT_TYPE_LATEST
from metrics import latest, REQUEST_TIME
from flask_httpauth import HTTPBasicAuth
from milvus import ping as milvus_ping
from minio_uploader import ping as minio_ping

blueprint = Blueprint("routes", __name__)

_session_not_found = "SESSION_NOT_FOUND"
_session_timed_out = "SESSION_TIMED_OUT"
_invalid_properties = "INVALID_PROPERTIES"
_rate_limit_exceeded = "RATE_LIMIT_EXCEEDED"
_rate_limit_negative_exceeded = "RATE_LIMIT_NEGATIVE_EXCEEDED"
_no_faces = "NO_FACES"
_no_primary_metadata = "NO_PRIMARY_METADATA"
_user_not_the_same = service._user_not_the_same
_user_disabled = "USER_DISABLED"
_already_uploaded = "ALREADY_UPLOADED"
_allowed_extensions = {'jpg', 'jpeg'}

_primary_photo_rate_limiter = MovingWindowRateLimiter(storage=MemoryStorage())
_primary_photo_rate_limiter_rate = None
def init_rate_limiters(app):
    global _primary_photo_rate_limiter_rate
    if app.config['PRIMARY_PHOTO_ERROR_LIMIT']:
        _primary_photo_rate_limiter_rate = parse(app.config['PRIMARY_PHOTO_ERROR_LIMIT'])
    else:
        _primary_photo_rate_limiter_rate = None

def _allowed_file_format(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in _allowed_extensions

@blueprint.route("/", methods = ["GET","POST"])
def home():
    if request.method == "POST":
        print(request.data)
        return str(request.data)
    return "<h1>Welcome to DeepFace API!</h1>"


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)

    obj = service.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1_path")
    img2_path = input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    distance_metric = input_args.get("distance_metric", "cosine")
    align = input_args.get("align", True)

    verification = service.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )

    verification["verified"] = str(verification["verified"])

    return verification


@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)
    actions = input_args.get("actions", ["age", "gender", "emotion", "race"])

    demographies = service.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    return demographies

@blueprint.route("/v1w/face-auth/similarity/<user_id>", methods=["POST"])
@auth_required
def similar(current_user, user_id):
    with REQUEST_TIME.labels(path="/v1w/face-auth/similarity").time():
        for img in request.files.getlist("image"):
            if _allowed_file_format(img.filename) is False:
                return {"message": "wrong image format", 'code': _invalid_properties}, 400

        try:
            bestIndex, euclidian, updateTime = service.check_similarity_and_update_secondary_photo(current_user, user_id, request.files.getlist("image"))
            return  {"userId":user_id, "bestIndex":bestIndex, "distance": euclidian, "secondaryPhotoUpdatedAt":updateTime}
        except exceptions.MetadataNotFound as e:
            _log_error(current_user, e)
            return {"message": str(e), "code":_no_primary_metadata}, 400
        except exceptions.NotSameUser as e:
            _log_error(current_user, e)
            return {"message": str(e), "code":_user_not_the_same}, 400
        except exceptions.NoFaces as e:
            _log_error(current_user, e)
            return {"message": str(e), "code":_no_faces}, 400
        except webhook.UnauthorizedFromWebhook as e:
            return str(e), 401
        except Exception as e:
            _log_error(current_user, e, True)
            return {"message":"oops, an error occured"}, 500
def _log_error(current_user: Token, e: Exception, unexpected = False):
    logging.error(f"[U:{current_user.user_id}] "+str(e), exc_info=e if unexpected else None)
@blueprint.route("/v1w/face-auth/primary_photo/<user_id>", methods=["POST"])
@auth_required
def primary_photo(current_user, user_id):
    with REQUEST_TIME.labels(path="/v1w/face-auth/primary_photo").time():
        if _allowed_file_format(request.files["image"].filename) is False:
            return {"message": "wrong image format", 'code': _invalid_properties}, 400

        try:
            if _primary_photo_rate_limiter_rate is not None and not _primary_photo_rate_limiter.test(_primary_photo_rate_limiter_rate, user_id): # global, should it be per user_id?
                return {"message": f"rate limit for errors {_primary_photo_rate_limiter_rate} exceeded", "code":_rate_limit_exceeded}, 429
            service.set_primary_photo(current_user, user_id, request.files["image"])
            return ""
        except exceptions.NoFaces as e:
            _log_error(current_user, e)
            if _primary_photo_rate_limiter_rate is not None:
                _primary_photo_rate_limiter.hit(_primary_photo_rate_limiter_rate, user_id)
            return {"message": str(e), "code":_no_faces}, 400
        except exceptions.MetadataAlreadyExists as e:
            _log_error(current_user, e)
            if _primary_photo_rate_limiter_rate is not None:
                _primary_photo_rate_limiter.hit(_primary_photo_rate_limiter_rate, user_id)
            return {"message": str(e), "code":_already_uploaded}, 409
        except exceptions.UserDisabled as e:
            _log_error(current_user, e)
            return {"message": str(e), "code":_user_disabled}, 403
        except webhook.UnauthorizedFromWebhook as e:
            return str(e), 401
        except Exception as e:
            _log_error(current_user, e, True)

            return {"message":"oops, an error occured"}, 500

@blueprint.route("/v1w/face-auth/", methods=["DELETE"])
@auth_required
def delete_photos(current_user: Token):
    with REQUEST_TIME.labels(path="/v1w/face-auth/").time():
        try:
            if current_app.config["MINIO_URI"]:
                service.delete_user_photos_and_metadata(current_user)
                return "", 200
            else:
                service.delete_temporary_user_data(current_user.user_id)
                return service.proxy_delete(current_user)
        except exceptions.MetadataNotFound as e:
            _log_error(current_user, e)
            return "", 204
        except webhook.UnauthorizedFromWebhook as e:
            return str(e), 401
        except Exception as e:
            _log_error(current_user, e, True)
            return {"message":"oops, an error occured"}, 500
@blueprint.route("/v1r/face-auth/status/<user_id>", methods=["GET"])
@auth_required
def user_status(current_user, user_id):
    with REQUEST_TIME.labels(path="/v1w/face-auth/status").time():
        try:
            status = service.get_status(user_id)
            return status
        except Exception as e:
            _log_error(current_user, e, True)
            return {"message":"oops, an error occured"}, 500

@blueprint.route("/v1w/face-auth/emotions/<user_id>", methods=["POST"])
@auth_required
def emotions(current_user, user_id):
    with REQUEST_TIME.labels(path="/v1w/face-auth/emotions").time():
        try:
            emotions_list, session_id, session_expired_at = service.emotions(user_id=user_id)

            return {'emotions': emotions_list, 'sessionId': session_id, 'sessionExpiredAt': session_expired_at}
        except exceptions.UserDisabled as e:
            _log_error(current_user, e)

            return {'message': str(e), 'code': _user_disabled}, 403
        except exceptions.RateLimitException as e:
            _log_error(current_user, e)

            return {'message': str(e), 'code': _rate_limit_exceeded}, 429
        except exceptions.NegativeRateLimitException as e:
            _log_error(current_user, e)

            return {'message': str(e), 'code': _rate_limit_negative_exceeded}, 429
        except Exception as e:
            _log_error(current_user, e)

            return {"message":"oops, an error occured"}, 500

@blueprint.route("/v1w/face-auth/liveness/<user_id>/<session_id>", methods=["POST"])
@auth_required
def liveness(current_user, user_id, session_id):
    with REQUEST_TIME.labels(path="/v1w/face-auth/liveness").time():
        images = request.files.getlist("image")
        if images is None:
            return {"message": "you must pass images input", 'code': _invalid_properties}, 400
        if len(images) != 15:
            return {"message": f"wrong number of images: {len(images)}", 'code': _invalid_properties}, 400
        for img in images:
            if _allowed_file_format(img.filename) is False:
                return {"message": "wrong image format", 'code': _invalid_properties}, 400

        try:
            result, session_ended, emotions = service.process_images(token=current_user.raw_token, user_id=user_id, session_id=session_id, images=images)

            return {'result': result, 'sessionEnded': session_ended, 'emotions': emotions.split(","), 'sessionId': session_id}
        except exceptions.WrongImageSizeException as e:
            _log_error(current_user, e)

            return {'message': str(e), 'code': _invalid_properties}, 400
        except exceptions.NoFaces as e:
            _log_error(current_user, e)

            return {"message": str(e), "code":_no_faces}, 400
        except exceptions.UserDisabled as e:
            _log_error(current_user, e)

            return {'message': str(e), 'code': _user_disabled}, 403
        except exceptions.SessionTimeOutException as e:
            _log_error(current_user, e)

            return {'message': str(e), 'code': _session_timed_out}, 403
        except exceptions.SessionNotFoundException as e:
            _log_error(current_user, e)

            return {'message': str(e), 'code': _session_not_found}, 404
        except exceptions.NegativeRateLimitException as e:
            _log_error(current_user, e)

            return {'message': str(e), 'code': _rate_limit_negative_exceeded}, 429
        except Exception as e:
            _log_error(current_user, e, True)

            return {"message":"oops, an error occured"}, 500

@blueprint.route("/v1w/face-auth/enable", methods=["POST"])
@auth_required
def enable_user(current_user: Token):
    if current_user.role != "admin":
        return {'message': f'insufficient role: "{current_user.role}"'}, 403
    data = request.get_json(force=True)
    if data is None:
        return {"message":"invalid json"}, 422
    user_id = data.get("userId",None)
    if not user_id:
        return {"message":"userId is missing"}, 422
    duplicated_face = data.get("duplicatedFace",None)
    try:
        service.reenable_user(current_user, user_id, duplicated_face)
        return "", 200
    except webhook.UnauthorizedFromWebhook as e:
        return str(e), 401
    except Exception as e:
        _log_error(current_user, e, True)

        return {"message": str(e)}, 500

metricsauth = HTTPBasicAuth()
@metricsauth.verify_password
def verify_password(username, password):
    return username == current_app.config["METRICS_USER"] and password == current_app.config['METRICS_PASSWORD']
@blueprint.route("/metrics")
@metricsauth.login_required
def metrics():
    if current_app.config['METRICS_PASSWORD'] == "":
        return "metrics disabled", 403
    data = latest()
    return Response(data, mimetype=CONTENT_TYPE_LATEST)
@blueprint.route("/health-check")
def healthcheck():
    timeout = 30
    try:
        milvus_ping()
        if current_app.config['MINIO_URI']:
            minio_ping()
        return "{}",200
    except Exception as e:
        logging.error(f"[health-check]: {str(e)}", exc_info=e)
        return str(e),500
