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


class SeedVR2WorkflowTests(unittest.TestCase):
    def test_short_upscale_uses_full_latent_path(self):
        workflow = BOT.build_seedvr2_workflow("input/test.mp4", 1920, "test/upscale")

        self.assertEqual(["6", 0], workflow["8"]["inputs"]["vae_conditioning"])
        self.assertEqual(["6", 0], workflow["10"]["inputs"]["latent_image"])
        self.assertEqual(["10", 0], workflow["12"]["inputs"]["samples"])

    def test_long_upscale_uses_temporal_split_and_merge(self):
        workflow = BOT.build_seedvr2_workflow(
            "input/test.mp4", 2560, "test/upscale", split_latent=True
        )

        self.assertEqual(["9", 0], workflow["8"]["inputs"]["vae_conditioning"])
        self.assertEqual(["9", 0], workflow["10"]["inputs"]["latent_image"])
        self.assertEqual(["11", 0], workflow["12"]["inputs"]["samples"])

    def test_upscale_dimensions_are_32_aligned(self):
        width, height = BOT.upscale_dimensions(736, 416, BOT.SEEDVR2_FHD_LONG_EDGE)

        self.assertEqual((1920, 1088), (width, height))
        self.assertEqual(0, width % 32)
        self.assertEqual(0, height % 32)


class SegmentedPromptTests(unittest.TestCase):
    def test_resolution_label_includes_megapixels(self):
        self.assertEqual("0.3 MP · 736×416", BOT.resolution_label(736, 416))
        self.assertEqual("0.5 MP · 960×544", BOT.resolution_label(960, 544))

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

    def test_later_segment_uses_first_audio_as_style_reference(self):
        job = make_job("GLOBAL:\nsame style\nSEGMENT 1:\nstart\nSEGMENT 2:\ncontinue", 2)
        job.audio_reference_name = "TelegramAudio/first_segment.mp4"
        rendered = BOT.segment_prompt(job)
        self.assertIn("<Audio 1>", rendered)
        self.assertIn("same music bed", rendered)

        workflow = BOT.build_workflow(
            job.config,
            rendered,
            output_prefix="test/audio",
            image_name="TelegramInputs/continuation.png",
            audio_reference_name=job.audio_reference_name,
        )
        conditioning = workflow["6"]["inputs"]
        self.assertEqual("auto", conditioning["task_type"])
        self.assertEqual("reference_only", conditioning["audio_mode"])
        self.assertTrue(conditioning["add_source_as_reference"])
        self.assertEqual(1, conditioning["prompt_primary_audio_ordinal"])
        audio_ref = conditioning["drive_audio"]
        self.assertEqual("LoadAudio", workflow[audio_ref[0]]["class_type"])

    def test_image_and_audio_references_do_not_overwrite_turbo_scheduler(self):
        config = BOT.GenerationConfig(864, 480, 8, 15.0, BOT.valid_length(15.0))
        workflow = BOT.build_workflow(
            config,
            "continue from the supplied image",
            image_name="TelegramInputs/continuation.png",
            audio_reference_name="TelegramAudio/previous.mp4",
        )

        self.assertEqual("BasicScheduler", workflow["13"]["class_type"])
        self.assertEqual(["13", 0], workflow["10"]["inputs"]["sigmas"])
        image_ref = workflow["6"]["inputs"]["first_frame"]
        audio_ref = workflow["6"]["inputs"]["drive_audio"]
        self.assertNotEqual(image_ref[0], "13")
        self.assertNotEqual(audio_ref[0], "13")
        self.assertEqual("LoadImage", workflow[image_ref[0]]["class_type"])
        self.assertEqual("LoadAudio", workflow[audio_ref[0]]["class_type"])


class TimelinePromptTests(unittest.TestCase):
    COMPLETE_TIMELINE = """【60秒反詐騙警示短片】

開頭（0-5秒）：
黑底警示標題，沉重低音音樂開始。

第一幕（5-15秒）：
同一名二十多歲女生在家看招聘廣告。
她猶豫後按下應聘按鈕，鏡頭特寫手指。

第二幕（15-25秒）：
她拖着行李抵達機場，露出期待笑容。
畫面逐漸轉為灰暗，節奏加快。

第三幕（25-40秒）：
接頭人收走護照和手機。
鐵閘關上，遠處有電網和高牆。
她驚慌站在角落。

第四幕（40-50秒）：
她被迫坐在電腦前輸入詐騙訊息。
她眼眶泛紅，神情空洞。

結尾（50-60秒）：
畫面轉黑，出現反詐騙警示。
音樂停止，只留下關門回響。
"""

    def test_chinese_timeline_is_split_into_short_ordered_shots(self):
        plan = BOT.build_long_video_plan(self.COMPLETE_TIMELINE, 60.0)

        self.assertEqual("timeline", plan.source_format)
        self.assertIn("60秒反詐騙警示短片", plan.global_text)
        self.assertEqual(11, len(plan.shots))
        self.assertAlmostEqual(0.0, plan.shots[0].start_seconds)
        self.assertAlmostEqual(60.0, plan.shots[-1].end_seconds)
        self.assertTrue(all(2.0 <= shot.duration <= 8.0 for shot in plan.shots))
        for previous, current in zip(plan.shots, plan.shots[1:]):
            self.assertAlmostEqual(previous.end_seconds, current.start_seconds)

    def test_timeline_missing_last_ten_seconds_fails_before_generation(self):
        incomplete = self.COMPLETE_TIMELINE.split("結尾（50-60秒）：", 1)[0]

        with self.assertRaisesRegex(BOT.BotError, "只寫到 50 秒"):
            BOT.build_long_video_plan(incomplete, 60.0)

    def test_plain_long_prompt_is_rejected_instead_of_replayed(self):
        with self.assertRaisesRegex(BOT.BotError, "必須提供時間軸"):
            BOT.build_long_video_plan("A woman walks through an airport.", 60.0)

    def test_planned_shot_prompt_contains_only_current_action(self):
        plan = BOT.build_long_video_plan(self.COMPLETE_TIMELINE, 60.0)
        job = make_job(self.COMPLETE_TIMELINE, segment_index=2, segment_total=len(plan.shots))
        job.total_seconds = 60.0
        job.shot_plan = plan.shots
        job.story_global_text = plan.global_text

        rendered = BOT.segment_prompt(job)

        self.assertIn("CURRENT SHOT ACTION", rendered)
        self.assertIn(plan.shots[1].action, rendered)
        self.assertNotIn(plan.shots[0].action, rendered)
        self.assertNotIn(plan.shots[2].action, rendered)
        self.assertIn("supplied first frame", rendered)

    def test_segment_format_is_also_split_into_short_shots(self):
        prompt = """GLOBAL:
same person and costume
SEGMENT 1:
action one. then action two.
SEGMENT 2:
action three. then action four.
SEGMENT 3:
action five. then action six.
SEGMENT 4:
action seven. then action eight.
"""
        plan = BOT.build_long_video_plan(prompt, 60.0)

        self.assertEqual("segments", plan.source_format)
        self.assertEqual(8, len(plan.shots))
        self.assertTrue(all(shot.duration <= 8.0 for shot in plan.shots))
        self.assertNotEqual(plan.shots[0].action, plan.shots[1].action)

    def test_segment_count_is_not_derived_from_fifteen_second_default(self):
        segments = "\n".join(
            f"SEGMENT {number}:\nunique action {number}."
            for number in range(1, 13)
        )
        plan = BOT.build_long_video_plan(
            f"GLOBAL:\nsame person and location\n{segments}",
            120.0,
        )

        self.assertEqual("segments", plan.source_format)
        # Twelve story SEGMENT blocks become two <=8-second H3 shots each
        # when the requested total is 120 seconds.
        self.assertEqual(24, len(plan.shots))
        self.assertAlmostEqual(0.0, plan.shots[0].start_seconds)
        self.assertAlmostEqual(120.0, plan.shots[-1].end_seconds)
        self.assertTrue(all(4.9 <= shot.duration <= 5.1 for shot in plan.shots))
        self.assertEqual("unique action 12.", plan.shots[-1].action)

    def test_segment_numbering_must_be_consecutive(self):
        prompt = "GLOBAL:\nsame style\nSEGMENT 1:\nstart\nSEGMENT 3:\nlater"

        with self.assertRaisesRegex(BOT.BotError, "expected SEGMENT 2"):
            BOT.build_long_video_plan(prompt, 30.0)

    def test_segment_count_cannot_make_each_segment_shorter_than_model_minimum(self):
        segments = "\n".join(
            f"SEGMENT {number}:\naction {number}."
            for number in range(1, 17)
        )

        with self.assertRaisesRegex(BOT.BotError, "too many"):
            BOT.build_long_video_plan(segments, 30.0)

    def test_transition_filter_preserves_story_offsets(self):
        shots = (
            BOT.ShotSpec(0.0, 5.0, "one", "first"),
            BOT.ShotSpec(5.0, 10.0, "two", "second"),
        )

        graph, video_output, audio_output = BOT.build_transition_filter(shots)

        self.assertIn("trim=duration=5.120", graph)
        self.assertIn("duration=0.120:offset=5.000", graph)
        self.assertIn("acrossfade=d=0.120", graph)
        self.assertEqual("[vx1]", video_output)
        self.assertEqual("[ax1]", audio_output)


if __name__ == "__main__":
    unittest.main()
