import importlib.util
import sys
import unittest
from pathlib import Path


BOT_PATH = Path(__file__).resolve().parents[1] / "MiniMax-H3-Telegram-Bot.py"
SPEC = importlib.util.spec_from_file_location("minimax_h3_generation_modes", BOT_PATH)
assert SPEC is not None and SPEC.loader is not None
BOT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BOT
SPEC.loader.exec_module(BOT)


class GenerationModeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = BOT.GenerationConfig(
            736,
            416,
            8,
            15.0,
            BOT.valid_length(15.0),
        )

    def test_fl2va_connects_first_and_last_frames(self):
        workflow = BOT.build_workflow(
            self.config,
            "continue the supplied shot",
            image_name="TelegramInputs/first.png",
            last_image_name="TelegramInputs/last.png",
            generation_mode=BOT.INPUT_MODE_FL2VA,
        )

        conditioning = workflow["6"]["inputs"]
        self.assertEqual("FL2VA", conditioning["task_type"])
        self.assertEqual("minimax_h3_fl2va_int8_convrot.safetensors", workflow["4"]["inputs"]["unet_name"])
        self.assertEqual("LoadImage", workflow[conditioning["first_frame"][0]]["class_type"])
        self.assertEqual("LoadImage", workflow[conditioning["last_frame"][0]]["class_type"])
        self.assertEqual("BasicScheduler", workflow["13"]["class_type"])
        self.assertEqual(["13", 0], workflow["10"]["inputs"]["sigmas"])

    def test_ref2va_connects_reference_images_videos_and_audio(self):
        workflow = BOT.build_workflow(
            self.config,
            "use the supplied references",
            reference_image_names=["TelegramInputs/ref.png"],
            reference_video_names=["TelegramInputs/ref.mp4"],
            reference_audio_names=["TelegramInputs/ref.wav"],
            generation_mode=BOT.INPUT_MODE_REF2VA,
        )

        conditioning = workflow["6"]["inputs"]
        self.assertEqual("Ref2VA", conditioning["task_type"])
        self.assertEqual(
            "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            workflow["4"]["inputs"]["unet_name"],
        )
        self.assertEqual(
            "LoadImage",
            workflow[conditioning["ref_images.ref_image_0"][0]]["class_type"],
        )
        components_id = conditioning["ref_videos.ref_video_0"][0]
        self.assertEqual("GetVideoComponents", workflow[components_id]["class_type"])
        self.assertEqual(
            components_id,
            conditioning["ref_video_audios.ref_video_audio_0"][0],
        )
        self.assertEqual(
            "LoadAudio",
            workflow[conditioning["ref_audios.ref_audio_0"][0]]["class_type"],
        )
        self.assertEqual("BasicScheduler", workflow["13"]["class_type"])

    def test_ref2va_without_audio_reference_uses_native_audio(self):
        workflow = BOT.build_workflow(
            self.config,
            "continue from the supplied visual reference",
            reference_image_names=["TelegramInputs/ref.png"],
            generation_mode=BOT.INPUT_MODE_REF2VA,
        )

        conditioning = workflow["6"]["inputs"]
        self.assertEqual("Ref2VA", conditioning["task_type"])
        self.assertEqual("native", conditioning["audio_mode"])
        self.assertFalse(conditioning["add_source_as_reference"])
        self.assertEqual(0, conditioning["prompt_primary_audio_ordinal"])
        self.assertNotIn("drive_audio", conditioning)

    def test_ref2va_requires_at_least_one_reference(self):
        with self.assertRaisesRegex(BOT.BotError, "至少需要"):
            BOT.build_workflow(
                self.config,
                "missing references",
                generation_mode=BOT.INPUT_MODE_REF2VA,
            )

    def test_mode_normalization_keeps_unknown_values_safe(self):
        self.assertEqual(BOT.INPUT_MODE_TEXT, BOT.normalize_input_mode("unknown"))
        self.assertEqual(BOT.INPUT_MODE_FL2VA, BOT.normalize_input_mode("FL2VA"))
        self.assertEqual(BOT.INPUT_MODE_REF2VA, BOT.normalize_input_mode("ref2va"))


if __name__ == "__main__":
    unittest.main()
