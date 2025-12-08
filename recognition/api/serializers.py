from django.contrib.auth import get_user_model

from rest_framework import serializers

from users.models import RecognitionAttempt, UserProfile

User = get_user_model()


class UserSerializer(serializers.ModelSerializer):
    """Serializer for the User model."""

    full_name = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = [
            "id",
            "username",
            "email",
            "first_name",
            "last_name",
            "full_name",
            "is_staff",
            "is_active",
        ]
        read_only_fields = ["is_staff", "is_active"]

    def get_full_name(self, obj):
        return obj.get_full_name()


class UserProfileSerializer(serializers.ModelSerializer):
    """Serializer for extended user profile data."""

    class Meta:
        model = UserProfile
        fields = ["department", "position", "phone_number"]


class EmployeeSerializer(UserSerializer):
    """Extended serializer for employees including profile data."""

    profile = UserProfileSerializer(source="userprofile", read_only=True)

    class Meta(UserSerializer.Meta):
        fields = UserSerializer.Meta.fields + ["profile"]


class RegisterEmployeeSerializer(serializers.ModelSerializer):
    """Serializer for registering new employees."""

    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ["username", "email", "password", "first_name", "last_name"]

    def create(self, validated_data):
        user = User.objects.create_user(**validated_data)
        return user


class AttendanceRecordSerializer(serializers.ModelSerializer):
    """Serializer for attendance records (RecognitionAttempt)."""

    username = serializers.SerializerMethodField()
    direction_display = serializers.CharField(source="get_direction_display", read_only=True)

    class Meta:
        model = RecognitionAttempt
        fields = [
            "id",
            "username",
            "direction",
            "direction_display",
            "confidence",
            "successful",
            "spoof_detected",
            "created_at",
            "source",
        ]
        read_only_fields = ["created_at", "username"]

    def get_username(self, obj):
        return obj.username or (obj.user.username if obj.user else "Unknown")


class RecognitionRequestSerializer(serializers.Serializer):
    """Serializer for face recognition requests."""

    image = serializers.CharField(help_text="Base64 encoded image data")
    direction = serializers.ChoiceField(choices=[("in", "In"), ("out", "Out")], required=False)


class StatsSerializer(serializers.Serializer):
    """Serializer for dashboard statistics."""

    total_employees = serializers.IntegerField()
    present_today = serializers.IntegerField()
    checked_out_today = serializers.IntegerField()
    pending_checkout = serializers.IntegerField()
