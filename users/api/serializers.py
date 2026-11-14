from rest_framework import serializers
from users.models import Profile, Report
from django.contrib.auth.models import User

class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ['id', 'username', 'email']

from django_countries.serializer_fields import CountryField

class ProfileSerializer(serializers.ModelSerializer):
    user = UserSerializer(read_only=True)
    country = CountryField(allow_blank=True)
    fast_avatar = serializers.SerializerMethodField()
    is_online = serializers.SerializerMethodField()

    class Meta:
        model = Profile
        fields = [
            'id', 'user', 'image', 'bio', 'gender', 'age', 'diamonds',
            'is_verified', 'psychological_profile', 'self_reported_traits',
            'connection_preferences', 'interests', 'expertise_areas',
            'conversation_topics', 'current_intent',
            'last_quiz_taken', 'is_profile_public', 'available_for_offline_search',
            'show_ai_analysis', 'hidden_data_fields',
            'country', 'state', 'languages', 'latitude', 'longitude',
            'persona_image_url', 'fast_avatar', 'trust_score', 'trust_tier',
            'subscription_tier', 'is_premium', 'subscription_expiry',
            'daily_ai_llm_searches', 'daily_standard_searches', 'daily_roulette_searches',
            'is_online',
        ]
        # SECURITY: anything that is paid for, earned, or computed by the server must be read-only.
        # Previously subscription_tier / subscription_expiry / quotas were writable via PATCH = free premium.
        read_only_fields = [
            'psychological_profile', 'last_quiz_taken', 'diamonds', 'is_verified',
            'conversation_topics', 'trust_score', 'trust_tier',
            'subscription_tier', 'is_premium', 'subscription_expiry',
            'daily_ai_llm_searches', 'daily_standard_searches', 'daily_roulette_searches',
            'current_intent', 'persona_image_url',
        ]

    # Fields that must never be shown to anyone except the profile owner.
    PRIVATE_FIELDS = (
        'latitude', 'longitude', 'subscription_tier', 'is_premium', 'subscription_expiry',
        'daily_ai_llm_searches', 'daily_standard_searches', 'daily_roulette_searches',
        'diamonds', 'current_intent', 'hidden_data_fields', 'connection_preferences',
        'self_reported_traits',
    )

    def get_is_online(self, instance):
        return instance.is_online()

    def get_fast_avatar(self, instance):
        """Uploaded photo → DiceBear (instant CDN). Never uses slow Pollinations URLs."""
        try:
            if instance.image and not str(instance.image).endswith('default.jpg'):
                return instance.image.url.replace('http://', 'https://')
        except Exception:
            pass
        return f'https://api.dicebear.com/7.x/adventurer/png?seed={instance.user.username}&size=200'

    def validate_image(self, value):
        # SECURITY: same validation/re-encoding as the dedicated upload endpoint.
        if not value:
            return value
        from users.api.views import sanitize_uploaded_image
        try:
            return sanitize_uploaded_image(value)
        except ValueError as e:
            raise serializers.ValidationError(str(e))

    def update(self, instance, validated_data):
        user_data = self.context['request'].data.get('user')
        if user_data and 'username' in user_data:
             user = instance.user
             new_username = user_data['username']
             if User.objects.filter(username=new_username).exclude(id=user.id).exists():
                 raise serializers.ValidationError({"username": "This username is already taken."})
             user.username = new_username
             user.save()
        
        return super().update(instance, validated_data)

    def to_representation(self, instance):
        data = super().to_representation(instance)
        
        request = self.context.get('request')
        is_owner = bool(request and request.user.is_authenticated and request.user == instance.user)
        if not is_owner:
            # SECURITY: never expose another user's email or exact location.
            public_user = {'id': instance.user.id, 'username': instance.user.username}
            if not instance.is_profile_public:
                return {'user': public_user, 'message': 'This profile is private.'}

            data['user'] = public_user
            for field in self.PRIVATE_FIELDS:
                data.pop(field, None)

            for field in (instance.hidden_data_fields or []):
                if field in data:
                    data.pop(field)
                    
            if not instance.show_ai_analysis:
                data.pop('psychological_profile', None)
                
        return data


class OnboardingQuizSerializer(serializers.Serializer):
    """
    Receives quiz answers covering psychology, interests, expertise, goals, and connection preferences.
    """
    answers = serializers.JSONField(help_text="Dictionary of question IDs and string answers.")
    connection_preferences = serializers.JSONField(help_text="What kind of people/conversations they seek.", required=False)
    interests = serializers.JSONField(help_text="List of topics they are interested in.", required=False)
    expertise = serializers.JSONField(help_text="List of topics they have expertise in.", required=False)

class ReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = Report
        fields = ['id', 'reported_user', 'reason', 'timestamp', 'status']
        read_only_fields = ['timestamp', 'status']
