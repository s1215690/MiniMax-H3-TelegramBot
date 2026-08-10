import importlib.util
import sys
import unittest
from pathlib import Path


BOT_PATH = Path(__file__).resolve().parents[1] / "MiniMax-H3-Telegram-Bot.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_telegram_bot_prompt_files", BOT_PATH)
assert SPEC is not None and SPEC.loader is not None
BOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOT
SPEC.loader.exec_module(BOT)


class PromptFileTests(unittest.TestCase):
    def test_decode_utf8_preserves_multiline_story_prompt(self):
        source = "GLOBAL:\n同一個人物和場景\n\nSEGMENT 1:\n鏡頭向前移動。\n"

        self.assertEqual(source.strip(), BOT.decode_prompt_text(source.encode("utf-8")))

    def test_decode_utf16_file(self):
        source = "長提示詞\n第二行"

        self.assertEqual(source, BOT.decode_prompt_text(source.encode("utf-16")))

    def test_prompt_file_info_accepts_txt(self):
        message = {
            "document": {
                "file_id": "file-1",
                "file_name": "story.txt",
                "mime_type": "application/octet-stream",
                "file_size": 123,
            }
        }

        self.assertEqual(
            ("file-1", "story.txt", 123),
            BOT.TelegramMenuBot.prompt_file_info(message),
        )

    def test_prompt_file_info_rejects_binary_document(self):
        message = {
            "document": {
                "file_id": "file-2",
                "file_name": "story.pdf",
                "mime_type": "application/pdf",
                "file_size": 123,
            }
        }

        self.assertIsNone(BOT.TelegramMenuBot.prompt_file_info(message))


if __name__ == "__main__":
    unittest.main()
