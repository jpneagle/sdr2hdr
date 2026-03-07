import unittest
from unittest import mock

from sdr2hdr.io import VideoInfo, build_audio_output_args, is_interlaced_video, open_decoder


class IoTests(unittest.TestCase):
    def test_is_interlaced_video_detects_progressive(self) -> None:
        info = VideoInfo(1920, 1080, 29.97, None, "yuv420p", 10.0, "progressive")
        self.assertFalse(is_interlaced_video(info))

    def test_is_interlaced_video_detects_interlaced_field_order(self) -> None:
        info = VideoInfo(1920, 1080, 29.97, None, "yuv420p", 10.0, "tt")
        self.assertTrue(is_interlaced_video(info))

    @mock.patch("sdr2hdr.io.subprocess.Popen")
    def test_open_decoder_inserts_bwdif_for_interlaced_input(self, popen_mock: mock.Mock) -> None:
        info = VideoInfo(1440, 1080, 29.97, None, "yuv420p", 10.0, "tb")
        open_decoder("input.m2ts", info)
        cmd = popen_mock.call_args.args[0]
        self.assertIn("-vf", cmd)
        self.assertIn("bwdif=mode=send_frame:parity=auto:deint=all", cmd)

    @mock.patch("sdr2hdr.io.subprocess.Popen")
    def test_open_decoder_skips_bwdif_for_progressive_input(self, popen_mock: mock.Mock) -> None:
        info = VideoInfo(1920, 1080, 23.976, None, "yuv420p", 10.0, "progressive")
        open_decoder("input.mp4", info)
        cmd = popen_mock.call_args.args[0]
        self.assertNotIn("-vf", cmd)

    @mock.patch("sdr2hdr.io.ffprobe_first_audio_codec", return_value="pcm_bluray")
    def test_build_audio_output_args_transcodes_pcm_bluray_for_mp4(self, _: mock.Mock) -> None:
        self.assertEqual(build_audio_output_args("output.mp4", "input.m2ts"), ["-c:a", "aac", "-b:a", "192k"])

    @mock.patch("sdr2hdr.io.ffprobe_first_audio_codec", return_value="aac")
    def test_build_audio_output_args_copies_supported_mp4_audio(self, _: mock.Mock) -> None:
        self.assertEqual(build_audio_output_args("output.mp4", "input.mp4"), ["-c:a", "copy"])

    @mock.patch("sdr2hdr.io.ffprobe_first_audio_codec", return_value=None)
    def test_build_audio_output_args_handles_missing_audio(self, _: mock.Mock) -> None:
        self.assertEqual(build_audio_output_args("output.mp4", "input.mp4"), [])


if __name__ == "__main__":
    unittest.main()
