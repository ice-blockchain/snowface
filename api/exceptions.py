class MetadataNotFound(Exception):
    def __init__(self, message):
        super().__init__(message)

class MetadataAlreadyExists(Exception):
    def __init__(self, message):
        super().__init__(message)

class NotSameUser(Exception):
    def __init__(self, message):
        super().__init__(message)

class UserDisabled(Exception):
    def __init__(self, message):
        super().__init__(message)

class UserNotFound(Exception):
    def __init__(self, message):
        super().__init__(message)

class NoFaces(Exception):
    def __init__(self, message):
        super().__init__(message)
class FailedTryToDisable(Exception):
    def __init__(self, message, sface_distance, arface_distance, matching_user_id, arcface_meta, similar_users):
        super().__init__(message)
        self.sface_distance = sface_distance
        self.arface_distance = arface_distance
        self.matching_user_id = matching_user_id
        self.arcface_metadata = arcface_meta
        self.similar_users = similar_users

class UpsertException(Exception):
    def __init__(self, message):
        super().__init__(message)

class SessionTimeOutException(Exception):
    def __init__(self, message):
        super().__init__(message)

class NoDataException(Exception):
    def __init__(self, message):
        super().__init__(message)

class WrongEmotionException(Exception):
    def __init__(self, message):
        super().__init__(message)

class SessionNotFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)

class RateLimitException(Exception):
    def __init__(self, message):
        super().__init__(message)
class NegativeRateLimitException(Exception):
    def __init__(self, message):
        super().__init__(message)

class WrongImageSizeException(Exception):
    def __init__(self, message):
        super().__init__(message)
