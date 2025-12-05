"""Context processors for recognition templates."""

from django.conf import settings


def multi_face_mode(request):
    """Add multi-face detection mode status to template context.

    This allows templates to display different UI based on whether
    single-person or multi-person recognition is enabled.

    Usage in templates:
        {% if multi_face_enabled %}
            <div class="alert alert-info">Group Mode Active</div>
        {% else %}
            <div class="alert alert-warning">Single Person Mode</div>
        {% endif %}
    """
    return {
        "multi_face_enabled": getattr(settings, "RECOGNITION_MULTI_FACE_ENABLED", False),
        "max_faces_per_frame": getattr(settings, "RECOGNITION_MAX_FACES_PER_FRAME", 5),
    }
