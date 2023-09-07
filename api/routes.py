from flask import Blueprint, request, Response
import service

from auth import auth_required
import traceback
blueprint = Blueprint("routes", __name__)

_no_faces = "NO_FACES"
_no_primary_metadata = "NO_PRIMARY_METADATA"
_user_not_the_same = "USER_NOT_THE_SAME"
_user_disabled = "USER_DISABLED"
_already_uploaded = "ALREADY_UPLOADED"

@blueprint.route("/")
def home():
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
        bestIndex, euclidian, updateTime = service.check_similar_user_and_register_metadata(user_id, request.files.getlist("image"))
        return  {"userID":user_id, "bestIndex":bestIndex, "distance": euclidian, "secondaryPhotoUpdatedAt":updateTime}
    except service.MetadataNotFound as e:
        return {"message": str(e), "code":_no_primary_metadata}, 400
    except service.NotSameUser as e:
        return {"message": str(e), "code":_user_not_the_same}, 400
    except service.NoFaces as e:
        return {"message": str(e), "code":_no_faces}, 400
    except Exception as e:
        print(traceback.format_exc())
        return {"message":"oops, an error occured"}, 500


@blueprint.route("/v1w/face-auth/primary_photo/<user_id>", methods=["POST"])
@auth_required
def primary_photo(current_user, user_id):
    try:
        service.set_primary_photo(user_id, request.files["image"])
        return ""
    except service.NoFaces as e:
        return {"message": str(e), "code":_no_faces}, 400
    except service.MetadataAlreadyExists as e:
        return {"message": str(e), "code":_already_uploaded}, 409
    except service.UserDisabled as e:
        return {"message": str(e), "code":_user_disabled}, 403

@blueprint.route("/status/<user_id>", methods=["GET"])
@auth_required
def user_status(current_user, user_id):
    status = service.get_status(user_id)
    return status

