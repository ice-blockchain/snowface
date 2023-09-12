from flask import Blueprint, request
import service, exceptions, webhook

from auth import auth_required
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter
from limits import parse
from webhook import UnauthorizedFromWebhook
import logging
blueprint = Blueprint("routes", __name__)

_session_not_found = "SESSION_NOT_FOUND"
_session_timed_out = "SESSION_TIMED_OUT"
_invalid_properties = "INVALID_PROPERTIES"
_rate_limit_exceeded = "RATE_LIMIT_EXCEEDED"
_rate_limit_negative_exceeded = "RATE_LIMIT_NEGATIVE_EXCEEDED"
_no_faces = "NO_FACES"
_no_primary_metadata = "NO_PRIMARY_METADATA"
_user_not_the_same = "USER_NOT_THE_SAME"
_user_disabled = "USER_DISABLED"
_already_uploaded = "ALREADY_UPLOADED"

_primary_photo_rate_limiter = MovingWindowRateLimiter(storage=MemoryStorage())
_primary_photo_rate_limiter_rate = None
def init_rate_limiters(app):
    global _primary_photo_rate_limiter_rate
    if app.config['PRIMARY_PHOTO_ERROR_LIMIT']:
        _primary_photo_rate_limiter_rate = parse(app.config['PRIMARY_PHOTO_ERROR_LIMIT'])
    else:
        _primary_photo_rate_limiter_rate = None

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
    try:
        bestIndex, euclidian, updateTime = service.check_similarity_and_update_secondary_photo(current_user, user_id, request.files.getlist("image"))
        return  {"userId":user_id, "bestIndex":bestIndex, "distance": euclidian, "secondaryPhotoUpdatedAt":updateTime}
    except exceptions.MetadataNotFound as e:
        logging.error(e)
        return {"message": str(e), "code":_no_primary_metadata}, 400
    except exceptions.NotSameUser as e:
        logging.error(e)
        return {"message": str(e), "code":_user_not_the_same}, 400
    except exceptions.NoFaces as e:
        logging.error(e)
        return {"message": str(e), "code":_no_faces}, 400
    except webhook.UnauthorizedFromWebhook as e:
        return str(e), 401
    except Exception as e:
        logging.error(e, exc_info=e)
        return {"message":"oops, an error occured"}, 500

@blueprint.route("/v1w/face-auth/primary_photo/<user_id>", methods=["POST"])
@auth_required
def primary_photo(current_user, user_id):
    try:
        if _primary_photo_rate_limiter_rate is not None and not _primary_photo_rate_limiter.test(_primary_photo_rate_limiter_rate, ""): # global, should it be per user_id?
            return {"message": f"rate limit for errors {_primary_photo_rate_limiter_rate} exceeded", "code":_rate_limit_exceeded}, 429
        service.set_primary_photo(current_user, user_id, request.files["image"])
        return ""
    except exceptions.NoFaces as e:
        logging.error(e)
        if _primary_photo_rate_limiter_rate is not None:
            _primary_photo_rate_limiter.hit(_primary_photo_rate_limiter_rate, "")
        return {"message": str(e), "code":_no_faces}, 400
    except exceptions.MetadataAlreadyExists as e:
        logging.error(e)
        if _primary_photo_rate_limiter_rate is not None:
            _primary_photo_rate_limiter.hit(_primary_photo_rate_limiter_rate, "")
        return {"message": str(e), "code":_already_uploaded}, 409
    except exceptions.UserDisabled as e:
        logging.error(e)
        return {"message": str(e), "code":_user_disabled}, 403
    except webhook.UnauthorizedFromWebhook as e:
        return str(e), 401
    except Exception as e:
        logging.error(e, exc_info=e)
        if _primary_photo_rate_limiter_rate is not None:
            _primary_photo_rate_limiter.hit(_primary_photo_rate_limiter_rate, "")
        return {"message":"oops, an error occured"}, 500

@blueprint.route("/status/<user_id>", methods=["GET"])
@auth_required
def user_status(current_user, user_id):
    try:
        status = service.get_status(user_id)
        return status
    except Exception as e:
        logging.error(e, exc_info=e)
        return {"message":"oops, an error occured"}, 500

@blueprint.route("/v1w/face-auth/emotions/<user_id>", methods=["POST"])
@auth_required
def emotions(current_user, user_id):
    try:
        emotions_list, session_id = service.emotions(user_id=user_id)

        return {'emotions': emotions_list, 'session_id': session_id}
    except exceptions.UserDisabled as e:
        return {'message': str(e), 'code': _user_disabled}, 403
    except exceptions.RateLimitException as e:
        return {'message': str(e), 'code': _rate_limit_exceeded}, 429
    except Exception as e:
        return {"message":"oops, an error occured"}, 500

@blueprint.route("/v1w/face-auth/emotions/<user_id>/<session_id>", methods=["PUT"])
@auth_required
def additional_emotion(current_user, user_id, session_id):
    try:
        emotions_list, session_id = service.add_additional_emotion(user_id=user_id, session_id=session_id)

        return {'emotions': emotions_list, 'sessionId': session_id}
    except exceptions.UserDisabled as e:
        logging.error(e)

        return {'message': str(e), 'code': _user_disabled}, 403
    except exceptions.SessionTimeOutException as e:
        logging.error(e)

        return {'message': str(e), 'code': _session_timed_out}, 403
    except exceptions.SessionNotFoundException as e:
        logging.error(e)

        return {'message': str(e), 'code': _session_not_found}, 404
    except exceptions.RateLimitException as e:
        logging.error(e)

        return {'message': str(e), 'code': _rate_limit_negative_exceeded}, 429
    except Exception as e:
        logging.error(e, exc_info=e)

        return {"message":"oops, an error occured"}, 500

@blueprint.route("/v1w/face-auth/liveness/<user_id>/<session_id>", methods=["POST"])
@auth_required
def liveness(current_user, user_id, session_id):
    images = request.files.getlist("image")
    if images is None:
        return {"message": "you must pass images input", 'code': _invalid_properties}, 400
    if len(images) != 15:
        return {"message": "wrong number of images", 'code': _invalid_properties}, 400

    try:
        result, session_ended = service.process_images(token=current_user.raw_token, user_id=user_id, session_id=session_id, images=images)

        return {'result': result, 'sessionEnded': session_ended}
    except exceptions.WrongImageSizeException as e:
        logging.error(e)

        return {'message': str(e), 'code': _invalid_properties}, 400
    except exceptions.UserDisabled as e:
        logging.error(e)

        return {'message': str(e), 'code': _user_disabled}, 403
    except exceptions.SessionTimeOutException as e:
        logging.error(e)

        return {'message': str(e), 'code': _session_timed_out}, 403
    except exceptions.SessionNotFoundException as e:
        logging.error(e)

        return {'message': str(e), 'code': _session_not_found}, 404
    except exceptions.RateLimitException as e:
        logging.error(e)

        return {'message': str(e), 'code': _rate_limit_negative_exceeded}, 429
    except Exception as e:
        logging.error(e, exc_info=e)

        return {"message":"oops, an error occured"}, 500
