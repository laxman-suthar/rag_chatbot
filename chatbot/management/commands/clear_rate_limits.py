from django.core.cache import cache
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Clear rate limit keys from Redis cache."

    def add_arguments(self, parser):
        parser.add_argument(
            "--pattern",
            default="*rate:*",
            help="Key pattern to delete (default: *rate:*).",
        )
        parser.add_argument(
            "--batch",
            type=int,
            default=500,
            help="Batch size for SCAN (default: 500).",
        )

    def handle(self, *args, **options):
        pattern = options["pattern"]
        batch = options["batch"]
        client = getattr(cache, "_cache", None)

        if client is None or not hasattr(client, "scan"):
            self.stderr.write(
                "Cache backend does not appear to be Redis. No keys cleared."
            )
            return

        cursor = 0
        total_deleted = 0
        while True:
            cursor, keys = client.scan(cursor=cursor, match=pattern, count=batch)
            if keys:
                total_deleted += client.delete(*keys)
            if cursor == 0:
                break

        self.stdout.write(
            self.style.SUCCESS(f"Cleared {total_deleted} key(s) matching {pattern}")
        )
