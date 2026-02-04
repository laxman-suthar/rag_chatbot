from django.conf import settings
from django.core.cache import cache
from django.http import JsonResponse

from .models import RequestLog


def get_client_ip(request):
    x_forwarded_for = request.META.get("HTTP_X_FORWARDED_FOR")
    if x_forwarded_for:
        return x_forwarded_for.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


class RateLimitAndLogMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response
        self.enabled = getattr(settings, "RATE_LIMIT_ENABLED", True)
        self.limit = getattr(settings, "RATE_LIMIT_REQUESTS_PER_HOUR", 10)
        self.window_seconds = getattr(settings, "RATE_LIMIT_WINDOW_SECONDS", 3)
        self.skip_paths = set(getattr(settings, "RATE_LIMIT_SKIP_PATHS", []))

    def __call__(self, request):
        ip_address = get_client_ip(request)
        user = request.user if request.user.is_authenticated else None
        user_key = f"user:{user.id}" if user else "anon"
        path = request.path or ""

        should_skip = any(path.startswith(prefix) for prefix in self.skip_paths)
        blocked = False

        if self.enabled and not should_skip:
            cache_key = f"rate:{ip_address}:{user_key}"
            count = cache.get(cache_key)
            if count is None:
                cache.set(cache_key, 1, timeout=self.window_seconds)
                count = 1
            else:
                count = cache.incr(cache_key)

            if count > self.limit:
                blocked = True

        if blocked:
            status_code = 429
            response = JsonResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status=status_code,
            )
            self._log_request(request, ip_address, user, status_code, blocked=True)
            return response

        response = self.get_response(request)
        status_code = getattr(response, "status_code", 200)
        self._log_request(request, ip_address, user, status_code, blocked=False)
        return response

    def _log_request(self, request, ip_address, user, status_code, blocked):
        try:
            RequestLog.objects.create(
                user=user,
                ip_address=ip_address,
                method=request.method,
                path=request.path or "",
                user_agent=request.META.get("HTTP_USER_AGENT", ""),
                status_code=status_code,
                was_blocked=blocked,
            )
        except Exception:
            # Do not break request flow if logging fails
            return
