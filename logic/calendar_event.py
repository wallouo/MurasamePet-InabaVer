"""Holiday / seasonal event lookup for MurasamePet-Inaba."""

import datetime
from typing import Optional, TypedDict


class HolidayEvent(TypedDict):
    name: str       # 日本語の節日名
    hint: str       # めぐるが使えるひとこと
    emotion: str    # happy / excited / neutral / tired


_HOLIDAYS: dict[tuple[int, int], HolidayEvent] = {
    (1,  1):  {"name": "お正月",           "hint": "あけおめ～！今年もよろしく！",                    "emotion": "excited"},
    (2,  3):  {"name": "節分",             "hint": "鬼は外～！豆まきした？",                         "emotion": "happy"},
    (2, 14):  {"name": "バレンタインデー",  "hint": "今日はバレンタインだし…べ、別に意識してないし！",   "emotion": "happy"},
    (3,  3):  {"name": "ひな祭り",         "hint": "ひな祭りだね～、ちらし寿司食べたい",               "emotion": "happy"},
    (3, 14):  {"name": "ホワイトデー",     "hint": "今日ホワイトデーじゃん、お返しどうする～？",        "emotion": "excited"},
    (4,  1):  {"name": "エイプリルフール", "hint": "今日だけは嘘ついてもOKっしょ！",                  "emotion": "excited"},
    (5,  5):  {"name": "こどもの日",       "hint": "こどもの日だよ～、柏餅食べた？",                  "emotion": "happy"},
    (7,  7):  {"name": "七夕",            "hint": "今日は七夕！お願いごと、もう決めた？",             "emotion": "happy"},
    (10, 31): {"name": "ハロウィン",       "hint": "ハロウィン～！仮装とかする？",                    "emotion": "excited"},
    (12, 24): {"name": "クリスマスイブ",   "hint": "クリスマスイブじゃん！今夜の予定は～？",           "emotion": "excited"},
    (12, 25): {"name": "クリスマス",       "hint": "メリクリ！サンタさん来た？",                      "emotion": "excited"},
    (12, 31): {"name": "大晦日",          "hint": "今年もあっという間だったね…来年もよろしく！",       "emotion": "happy"},
}


def get_today_holiday() -> Optional[HolidayEvent]:
    """Return today's holiday event dict, or None if not a special day."""
    today = datetime.date.today()
    return _HOLIDAYS.get((today.month, today.day))


def get_holiday_hint() -> Optional[str]:
    """Convenience: return only the hint string for prompt injection."""
    event = get_today_holiday()
    return event["hint"] if event else None


if __name__ == "__main__":
    event = get_today_holiday()
    if event:
        print(f"今日: {event['name']}")
        print(f"ひとこと: {event['hint']}")
        print(f"感情: {event['emotion']}")
    else:
        print("今日は特別なイベントはないよ。")