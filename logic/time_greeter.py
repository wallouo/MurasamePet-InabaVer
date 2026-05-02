"""Time-based greeting generator for MurasamePet-Inaba."""

from datetime import datetime
from typing import TypedDict


class TimeGreeting(TypedDict):
    text: str       # めぐるのセリフ（日本語）
    emotion: str    # happy / neutral / tired / excited


def get_time_greeting() -> TimeGreeting:
    """Return a time-appropriate greeting from めぐる."""
    hour = datetime.now().hour

    if 5 <= hour < 9:
        return {"text": "おはよ～！早起きじゃん、えらい！今日も一緒に頑張ろ！", "emotion": "happy"}
    elif 9 <= hour < 12:
        return {"text": "午前中だし、まだまだ元気出していこっ！",               "emotion": "happy"}
    elif 12 <= hour < 14:
        return {"text": "お昼だよ～！ちゃんとごはん食べた？サボっちゃダメだよ？", "emotion": "neutral"}
    elif 14 <= hour < 18:
        return {"text": "午後もがんばってるじゃん！休憩したくなったら言って？",  "emotion": "neutral"}
    elif 18 <= hour < 21:
        return {"text": "お疲れ様～！今日もよく頑張ったじゃん、えらい！",       "emotion": "happy"}
    elif 21 <= hour < 24:
        return {"text": "夜だね…無理しすぎてない？ちゃんと休んでよ？",          "emotion": "tired"}
    else:  # 0 <= hour < 5
        return {"text": "こんな時間まで起きてるの！？センパイ、ちゃんと寝なよ！？", "emotion": "tired"}


if __name__ == "__main__":
    result = get_time_greeting()
    print(f"セリフ : {result['text']}")
    print(f"感情   : {result['emotion']}")