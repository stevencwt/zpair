import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

# =========================
# USER SETTINGS
# =========================
SCHEDULE_FILE = "config_schedule.json"     # weekly schedule JSON
VMMAIN_URL = "http://localhost:8090"  # vmain control endpoint (POST {action:...})
TZ_NAME = "Asia/Singapore"

# How often to wake up and check time
TICK_SECONDS = 5

# Act only in the first N seconds of a new hour (to avoid missed boundary due to tick)
BOUNDARY_WINDOW_SECONDS = 45

# Reliability: send command multiple times at boundary / startup / resume-on-change
SEND_COUNT = 3
SEND_SPACING_SECONDS = 10

# Reload schedule periodically so edits to JSON are picked up without restart
RELOAD_SCHEDULE_EVERY_SECONDS = 60

# NEW: If schedule changes and it now says RESUME for the current hour, enforce immediately.
# (Safe: does not "pause you mid-hour", only resumes mid-hour.)
IMMEDIATE_RESUME_ON_SCHEDULE_CHANGE = True


# =========================
# INTERNALS
# =========================
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
TZ = ZoneInfo(TZ_NAME)


@dataclass
class Schedule:
    enabled: bool
    weekly: dict  # day -> [bool]*24


def load_schedule(path: str) -> Schedule:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Schedule file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    enabled = bool(data.get("enabled", True))
    weekly_schedule = data.get("weekly_schedule", None)
    if not isinstance(weekly_schedule, dict):
        raise ValueError("Invalid schedule JSON: missing or invalid 'weekly_schedule' object")

    weekly = {}
    for d in DAYS:
        day_obj = weekly_schedule.get(d, {})
        hours = day_obj.get("soft_pause_hours", None)
        if not isinstance(hours, list) or len(hours) != 24:
            raise ValueError(f"Invalid schedule JSON: {d}.soft_pause_hours must be a list of 24 booleans")
        weekly[d] = [bool(x) for x in hours]

    return Schedule(enabled=enabled, weekly=weekly)


def sg_now() -> datetime:
    return datetime.now(TZ)


def sg_day_key(dt: datetime) -> str:
    # Monday=0 ... Sunday=6
    return DAYS[dt.weekday()]


def desired_action(schedule: Schedule, dt: datetime) -> str:
    """
    If schedule disabled: always resume.
    Else: use today's hour boolean.
    """
    if not schedule.enabled:
        return "RESUME_TRADING"
    day = sg_day_key(dt)
    hour = dt.hour
    should_pause = schedule.weekly[day][hour]
    return "SET_SOFT_PAUSE" if should_pause else "RESUME_TRADING"


def send_action(action: str, url: str) -> bool:
    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        json={"action": action},
        timeout=5,
    )
    return 200 <= resp.status_code < 300


def send_with_retries(action: str, url: str, reason: str):
    """
    Send the command multiple times to improve reliability.
    """
    stamp = sg_now().isoformat()
    print(f"[{stamp}] SEND {action} x{SEND_COUNT} ({reason})")

    for i in range(SEND_COUNT):
        ok = send_action(action, url)
        stamp2 = sg_now().isoformat()
        print(f"[{stamp2}]   -> {action} ({i+1}/{SEND_COUNT}) ok={ok}")
        if i < SEND_COUNT - 1:
            time.sleep(SEND_SPACING_SECONDS)


def in_boundary_window(dt: datetime) -> bool:
    return dt.minute == 0 and dt.second <= BOUNDARY_WINDOW_SECONDS


def compress_pause_ranges(hours: list[bool]) -> list[tuple[int, int]]:
    """
    Convert 24 booleans into [(start_hour, end_hour)] ranges, end exclusive.
    Example: pause at 21,22 => [(21,23)] meaning 21:00–23:00.
    """
    ranges: list[tuple[int, int]] = []
    start = None

    for h in range(25):  # sentinel at 24
        cur = hours[h] if h < 24 else False
        if cur and start is None:
            start = h
        elif not cur and start is not None:
            ranges.append((start, h))
            start = None
    return ranges


def format_ranges(ranges: list[tuple[int, int]]) -> str:
    def hhmm(h: int) -> str:
        return f"{h:02d}:00"
    return ", ".join(f"{hhmm(s)}–{hhmm(e)}" for s, e in ranges)


def print_today_schedule(schedule: Schedule, dt: datetime):
    day = sg_day_key(dt)
    hours = schedule.weekly[day]
    ranges = compress_pause_ranges(hours)

    day_title = day.capitalize()
    if not schedule.enabled:
        print(f"[INFO] {day_title} schedule (SG): Scheduler disabled (will RESUME_TRADING always)")
        return

    if not ranges:
        print(f"[INFO] {day_title} schedule (SG): No soft pause today (trading allowed all hours)")
    else:
        print(f"[INFO] {day_title} schedule (SG): Soft pause will be enforced during {format_ranges(ranges)}")


def safe_get_mtime(path: str) -> float | None:
    try:
        return os.path.getmtime(path)
    except Exception:
        return None


def main():
    print(f"Schedule enforcer starting (TZ={TZ_NAME})")
    print(f"Reading schedule: {SCHEDULE_FILE}")
    print(f"vmain URL: {VMMAIN_URL}")
    print(f"Boundary window: first {BOUNDARY_WINDOW_SECONDS}s of each hour")
    print(f"Send count: {SEND_COUNT} (spacing {SEND_SPACING_SECONDS}s)")
    print(f"Immediate resume on schedule change: {IMMEDIATE_RESUME_ON_SCHEDULE_CHANGE}")
    print("-----")

    schedule = load_schedule(SCHEDULE_FILE)
    last_schedule_load_ts = time.time()
    last_mtime = safe_get_mtime(SCHEDULE_FILE)

    # Track day-change for printing schedule
    dt0 = sg_now()
    last_day_key = sg_day_key(dt0)
    print_today_schedule(schedule, dt0)

    # Track boundary enforcement + last action sent
    last_processed_boundary_key = None  # e.g. "2026-01-10 monday 21"
    last_sent_action = None

    # STARTUP ENFORCEMENT: enforce current hour immediately
    startup_desired = desired_action(schedule, dt0)
    send_with_retries(startup_desired, VMMAIN_URL, reason="startup immediate enforce")
    last_sent_action = startup_desired

    while True:
        try:
            now_ts = time.time()

            # Periodic schedule reload
            if now_ts - last_schedule_load_ts >= RELOAD_SCHEDULE_EVERY_SECONDS:
                current_mtime = safe_get_mtime(SCHEDULE_FILE)
                schedule_changed = (current_mtime is not None and current_mtime != last_mtime)

                schedule = load_schedule(SCHEDULE_FILE)
                last_schedule_load_ts = now_ts
                last_mtime = current_mtime

                print(f"[{sg_now().isoformat()}] Reloaded schedule file (changed={schedule_changed})")

                # NEW: if schedule changed and it now says RESUME for current hour, enforce immediately
                if schedule_changed:
                    dt_now = sg_now()
                    desired_now = desired_action(schedule, dt_now)

                    print(f"[{dt_now.isoformat()}] After reload: desired_now={desired_now}, last_sent_action={last_sent_action}")

                    # Enforce immediately on schedule change (both directions)
                    if desired_now != last_sent_action:
                        if desired_now == "RESUME_TRADING":
                            send_with_retries("RESUME_TRADING", VMMAIN_URL, reason="schedule updated -> resume now")
                            last_sent_action = "RESUME_TRADING"
                        elif desired_now == "SET_SOFT_PAUSE":
                            send_with_retries("SET_SOFT_PAUSE", VMMAIN_URL, reason="schedule updated -> pause now")
                            last_sent_action = "SET_SOFT_PAUSE"

            dt = sg_now()
            day_key = sg_day_key(dt)

            # Print today's schedule if day changes (midnight)
            if day_key != last_day_key:
                last_day_key = day_key
                print_today_schedule(schedule, dt)

                # Enforce immediately at day change if it implies a change
                desired = desired_action(schedule, dt)
                if desired != last_sent_action:
                    send_with_retries(desired, VMMAIN_URL, reason="day change immediate enforce")
                    last_sent_action = desired

            # Enforce at hour boundary only (primary policy)
            if in_boundary_window(dt):
                boundary_key = f"{dt.date().isoformat()} {day_key} {dt.hour:02d}"

                if boundary_key != last_processed_boundary_key:
                    last_processed_boundary_key = boundary_key

                    desired = desired_action(schedule, dt)

                    # Manual-override friendliness:
                    # Only send if desired action changed vs last sent action.
                    if desired != last_sent_action:
                        send_with_retries(desired, VMMAIN_URL, reason=f"hour boundary {dt.hour:02d}:00")
                        last_sent_action = desired
                    else:
                        print(f"[{dt.isoformat()}] Boundary {boundary_key}: desired={desired} (unchanged) -> not sent")

            time.sleep(TICK_SECONDS)

        except KeyboardInterrupt:
            print("Exiting schedule enforcer (Ctrl+C).")
            return
        except Exception as e:
            print(f"[{sg_now().isoformat()}] ERROR: {e}")
            time.sleep(5)


if __name__ == "__main__":
    main()