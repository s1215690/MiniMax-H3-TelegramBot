import importlib.util
import sys
import unittest
from pathlib import Path


BOT_PATH = Path(__file__).resolve().parents[1] / "MiniMax-H3-Telegram-Bot.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_telegram_bot", BOT_PATH)
assert SPEC is not None and SPEC.loader is not None
BOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOT
SPEC.loader.exec_module(BOT)


def make_job(prompt: str, segment_index: int, segment_total: int = 2):
    config = BOT.GenerationConfig(448, 256, 12, 15.0, BOT.valid_length(15.0))
    return BOT.JobState(
        chat_id="test",
        config=config,
        prompt=prompt,
        started_at=0.0,
        segment_index=segment_index,
        segment_total=segment_total,
        total_seconds=segment_total * 15.0,
    )


class SegmentedPromptTests(unittest.TestCase):
    def test_current_segment_isolated_from_other_actions(self):
        prompt = """GLOBAL:
same person, costume, room and daylight

SEGMENT 1:
opening action

SEGMENT 2:
continuation action

SEGMENT 3:
later action

---
same camera language and realistic style
"""
        result = BOT.segment_prompt(make_job(prompt, segment_index=2))

        self.assertIn("same person, costume, room and daylight", result)
        self.assertIn("same camera language and realistic style", result)
        self.assertIn("continuation action", result)
        self.assertNotIn("opening action", result)
        self.assertNotIn("later action", result)
        self.assertNotIn("same fight", result)

    def test_fullwidth_colon_and_inline_text_are_supported(self):
        parsed = BOT.parse_segmented_prompt(
            "GLOBAL： shared identity\nSEGMENT 1： first action\nSEGMENT 2: second action"
        )

        self.assertIsNotNone(parsed)
        self.assertEqual("shared identity", parsed.global_text)
        self.assertEqual("first action", parsed.segments[1])
        self.assertEqual("second action", parsed.segments[2])

    def test_missing_segment_fails_instead_of_replaying_another_segment(self):
        prompt = "GLOBAL:\nshared\nSEGMENT 1:\nonly opening"

        with self.assertRaisesRegex(BOT.BotError, "SEGMENT 2"):
            BOT.segment_prompt(make_job(prompt, segment_index=2))

    def test_plain_prompt_keeps_backward_compatible_long_video_behavior(self):
        prompt = "A continuous walk through a bright market."
        result = BOT.segment_prompt(make_job(prompt, segment_index=2))

        self.assertIn(prompt, result)
        self.assertIn("supplied first frame", result)
        self.assertNotIn("same fight", result)


if __name__ == "__main__":
    unittest.main()
