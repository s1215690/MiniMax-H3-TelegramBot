import importlib.util
import sys
import unittest
from pathlib import Path


BOT_PATH = Path(__file__).resolve().parents[1] / "MiniMax-H3-Telegram-Bot.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_telegram_bot_vram", BOT_PATH)
assert SPEC is not None and SPEC.loader is not None
BOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOT
SPEC.loader.exec_module(BOT)


class VramFallbackTests(unittest.TestCase):
    def test_resolution_fallback_descends_by_one_preset(self):
        self.assertEqual((736, 416), BOT.next_lower_resolution(864, 480))
        self.assertEqual((608, 352), BOT.next_lower_resolution(736, 416))
        self.assertEqual((512, 288), BOT.next_lower_resolution(608, 352))
        self.assertIsNone(BOT.next_lower_resolution(448, 256))

    def test_custom_resolution_uses_next_lower_preset(self):
        self.assertEqual((864, 480), BOT.next_lower_resolution(900, 500))

    def test_only_memory_errors_trigger_fallback(self):
        self.assertTrue(
            BOT.is_cuda_oom_error(
                RuntimeError("Allocation on device 0 would exceed allowed memory")
            )
        )
        self.assertTrue(BOT.is_cuda_oom_error(RuntimeError("CUDA out of memory")))
        self.assertFalse(BOT.is_cuda_oom_error(RuntimeError("connection reset")))

    def test_transition_filter_normalizes_output_size(self):
        shots = (BOT.ShotSpec(0.0, 5.0, "a", "first"), BOT.ShotSpec(5.0, 10.0, "b", "second"))

        graph, video_output, audio_output = BOT.build_transition_filter(
            shots,
            output_size=(864, 480),
        )

        self.assertIn("scale=864:480", graph)
        self.assertIn("pad=864:480", graph)
        self.assertEqual("[vx1]", video_output)
        self.assertEqual("[ax1]", audio_output)


if __name__ == "__main__":
    unittest.main()
