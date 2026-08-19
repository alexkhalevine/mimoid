import sqlite3
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

from . import config, db

NEXTCLOUD_BACKUP_DIR = "Mimoid"


class NextcloudError(Exception):
    """Raised when the Nextcloud upload fails, with a user-facing message."""


def _normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")
    return url


def get_backup_config() -> dict:
    url = db.get_setting("backup_nextcloud_url", "")
    username = db.get_setting("backup_nextcloud_username", "")
    password = db.get_setting("backup_nextcloud_password", "")
    last_backup_at = db.get_setting("backup_last_at", "")
    return {
        "url": url,
        "username": username,
        "has_password": bool(password),
        "last_backup_at": last_backup_at or None,
    }


def set_backup_config(url: str, username: str, password: str | None) -> dict:
    db.set_setting("backup_nextcloud_url", _normalize_url(url))
    db.set_setting("backup_nextcloud_username", username.strip())
    if password:
        db.set_setting("backup_nextcloud_password", password)
    return get_backup_config()


def create_backup_archive(dest: Path) -> int:
    """Builds a zip of a consistent SQLite snapshot + voice sample wavs at
    `dest`. Synchronous (uses blocking sqlite3/zipfile calls) -- callers
    should run this in a threadpool. Returns the archive size in bytes."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_snapshot = Path(tmp_dir) / "mimoid.db"
        source = sqlite3.connect(config.DB_PATH)
        try:
            snapshot = sqlite3.connect(db_snapshot)
            try:
                source.backup(snapshot)
            finally:
                snapshot.close()
        finally:
            source.close()

        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(db_snapshot, "mimoid.db")
            for wav_path in sorted(config.VOICE_SAMPLES_DIR.glob("*.wav")):
                archive.write(wav_path, f"voice_samples/{wav_path.name}")
            info = (
                "Mimoid backup\n"
                f"Created: {datetime.now(UTC).isoformat()}\n\n"
                "Contains: mimoid.db (conversations, memories, voice sample\n"
                "metadata, style corpus, settings) and voice_samples/ (reference\n"
                "audio for voice cloning). The ChromaDB memory index is not\n"
                "included -- it's derived from mimoid.db and can be rebuilt\n"
                "by re-indexing memories. There is no automated restore tool yet;\n"
                "restoring means replacing sidecar/data/mimoid.db and\n"
                "sidecar/data/voice_samples/ with the contents of this archive.\n"
            )
            archive.writestr("backup-info.txt", info)

    return dest.stat().st_size


def _map_response_error(response: httpx.Response, action: str) -> NextcloudError:
    if response.status_code in (401, 403):
        return NextcloudError(
            "Nextcloud rejected the credentials. If two-factor authentication is "
            "enabled on this Nextcloud account, your regular account password will "
            "never work here -- generate an app password instead: Nextcloud web UI "
            "-> Settings -> Security -> \"Devices & Sessions\" -> \"Create new app "
            "password\", then paste that (not your login password) into the "
            "password field above."
        )
    if response.status_code == 404:
        return NextcloudError(
            "Nextcloud path not found -- is the base URL correct? "
            "(expected the Nextcloud root, e.g. https://cloud.example.com)"
        )
    return NextcloudError(f"Nextcloud {action} failed with status {response.status_code}.")


async def upload_backup(archive: Path, url: str, username: str, password: str) -> str:
    """Uploads `archive` to the user's Nextcloud via WebDAV, creating the
    backup folder if needed. Returns the remote path on success."""
    remote_dir = f"{url}/remote.php/dav/files/{username}/{NEXTCLOUD_BACKUP_DIR}"
    remote_path = f"{remote_dir}/{archive.name}"

    try:
        async with httpx.AsyncClient(
            auth=httpx.BasicAuth(username, password),
            timeout=httpx.Timeout(30.0, read=300.0),
        ) as client:
            mkcol_response = await client.request("MKCOL", remote_dir)
            # 201 = created, 405 = already exists -- both are fine. Any other
            # non-2xx/405 (401/403/404/...) is a real problem, reported below.
            if mkcol_response.status_code not in (201, 405):
                raise _map_response_error(mkcol_response, "folder creation")

            put_response = await client.put(remote_path, content=archive.read_bytes())
            if not put_response.is_success:
                raise _map_response_error(put_response, "upload")
    except httpx.TimeoutException:
        raise NextcloudError(f"Timed out reaching Nextcloud at {url}.") from None
    except httpx.ConnectError:
        raise NextcloudError(
            f"Could not reach Nextcloud at {url}. Check the URL and your network."
        ) from None
    except httpx.HTTPError as err:
        raise NextcloudError(f"Nextcloud request failed: {err}") from None

    return remote_path
