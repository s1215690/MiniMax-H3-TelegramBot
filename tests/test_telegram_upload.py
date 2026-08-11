import importlib.util
import sys
import unittest
from pathlib import Path


BOT_PATH = Path(__file__).resolve().parents[1] / "MiniMax-H3-Telegram-Bot.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_telegram_bot_upload", BOT_PATH)
assert SPEC is not None and SPEC.loader is not None
BOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOT
SPEC.loader.exec_module(BOT)


class TelegramUploadTests(unittest.TestCase):
    def test_safe_limit_leaves_request_margin(self):
        self.assertLess(BOT.TELEGRAM_SAFE_VIDEO_BYTES, BOT.TELEGRAM_MAX_VIDEO_BYTES)
        self.assertLessEqual(BOT.TELEGRAM_SAFE_VIDEO_BYTES, 48_000_000)

    def test_target_bitrate_is_positive_and_duration_aware(self):
        self.assertGreater(BOT.telegram_video_target_bitrate(300), 256)
        self.assertGreater(
            BOT.telegram_video_target_bitrate(300),
            BOT.telegram_video_target_bitrate(600),
        )
        self.assertEqual(BOT.telegram_video_target_bitrate(300, 0.0), 256)

    def test_caption_note_is_bounded(self):
        caption = BOT.telegram_caption_with_note("x" * 2000, "compressed")
        self.assertLessEqual(len(caption), 1024)
        self.assertTrue(caption.endswith("..."))


if __name__ == "__main__":
    unittest.main()
