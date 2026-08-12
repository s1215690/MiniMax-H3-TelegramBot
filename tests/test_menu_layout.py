import importlib.util
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock


BOT_PATH = Path(__file__).resolve().parents[1] / "MiniMax-H3-Telegram-Bot.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_menu_layout", BOT_PATH)
assert SPEC is not None and SPEC.loader is not None
BOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOT
SPEC.loader.exec_module(BOT)


def callback_data(markup):
    return [
        button["callback_data"]
        for row in markup["inline_keyboard"]
        for button in row
    ]


class MenuLayoutTests(unittest.TestCase):
    def setUp(self):
        self.bot = BOT.TelegramMenuBot.__new__(BOT.TelegramMenuBot)
        self.bot.settings = BOT.GenerationConfig(
            736,
            416,
            8,
            15.0,
            BOT.valid_length(15.0),
        )
        self.bot.total_seconds = 15.0
        self.bot.input_mode = BOT.INPUT_MODE_TEXT
        self.bot.reference_image_paths = []
        self.bot.reference_video_paths = []
        self.bot.reference_audio_paths = []
        self.bot.shutdown_after_generation = False
        self.bot._shutdown_pending = False
        self.bot.lock = threading.RLock()
        self.bot.job = None

    def test_main_menu_is_compact_and_links_to_sections(self):
        data = callback_data(self.bot.menu_markup(BOT.MENU_MAIN))
        self.assertEqual(7, len(data))
        self.assertIn("generate", data)
        self.assertIn("menu:input", data)
        self.assertIn("menu:settings", data)
        self.assertIn("menu:job", data)
        self.assertIn("menu:system", data)
        self.assertIn("menu:history", data)
        self.assertNotIn("res:736x416", data)

    def test_settings_sections_keep_existing_controls(self):
        settings = callback_data(self.bot.menu_markup(BOT.MENU_SETTINGS))
        self.assertEqual(
            ["menu:mode", "menu:duration", "menu:quality", "last", "menu:main"],
            settings,
        )

        mode = callback_data(self.bot.menu_markup(BOT.MENU_MODE))
        self.assertIn("mode:text", mode)
        self.assertIn("mode:image", mode)
        self.assertIn("mode:fl2va", mode)
        self.assertIn("mode:ref2va", mode)

        duration = callback_data(self.bot.menu_markup(BOT.MENU_DURATION))
        self.assertIn("sec:15", duration)
        self.assertIn("sec:1800", duration)
        self.assertIn("sec_custom", duration)

        quality = callback_data(self.bot.menu_markup(BOT.MENU_QUALITY))
        self.assertIn("res:736x416", quality)
        self.assertIn("steps:4", quality)
        self.assertIn("steps:8", quality)
        self.assertIn("steps:12", quality)

    def test_job_and_system_controls_are_in_submenus(self):
        job = callback_data(self.bot.menu_markup(BOT.MENU_JOB))
        self.assertEqual(["progress", "noop", "menu:main"], job)

        system = callback_data(self.bot.menu_markup(BOT.MENU_SYSTEM))
        self.assertIn("temperature", system)
        self.assertIn("comfy_start", system)
        self.assertIn("comfy_status", system)
        self.assertIn("comfy_restart", system)
        self.assertIn("comfy_stop", system)
        self.assertIn("bot_restart", system)
        self.assertIn("shutdown_toggle", system)
        self.assertEqual("menu:main", system[-1])

    def test_menu_callback_changes_section_without_touching_generation(self):
        self.bot.allowed_chat_id = "123"
        self.bot.telegram = Mock()
        self.bot.show_menu = Mock()

        self.bot.handle_callback(
            {
                "id": "query-1",
                "data": "menu:quality",
                "message": {"chat": {"id": "123"}, "message_id": 77},
            }
        )

        self.assertEqual(BOT.MENU_QUALITY, self.bot.menu_section)
        self.bot.show_menu.assert_called_once_with(
            "123", 77, section=BOT.MENU_QUALITY
        )


if __name__ == "__main__":
    unittest.main()
