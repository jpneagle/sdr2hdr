import unittest
from unittest import mock

from sdr2hdr.io import (
    VideoInfo,
    build_audio_output_args,
    build_hdr_scale_filter,
    is_interlaced_video,
    open_decoder,
    open_encoder,
)


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

    def test_build_hdr_scale_filter_resizes_with_color_matrix(self) -> None:
        self.assertEqual(
            build_hdr_scale_filter(1920, 1080, 3840, 2160, scaler="lanczos"),
            "scale=3840:2160:flags=lanczos:in_color_matrix=bt2020:out_color_matrix=bt2020",
        )

    @mock.patch("sdr2hdr.io.ffprobe_first_audio_codec", return_value=None)
    @mock.patch("sdr2hdr.io.subprocess.Popen")
    def test_open_encoder_adds_resize_filter_for_nvenc(
        self,
        popen_mock: mock.Mock,
        _: mock.Mock,
    ) -> None:
        info = VideoInfo(1920, 1080, 23.976, None, "yuv420p", 10.0, "progressive")
        open_encoder(
            "output.mp4",
            "input.mp4",
            info,
            peak_nits=1000.0,
            encoder="hevc_nvenc",
            output_width=3840,
            output_height=2160,
            scaler="lanczos",
        )
        cmd = popen_mock.call_args.args[0]
        self.assertEqual(cmd[cmd.index("-s") + 1], "1920x1080")
        self.assertIn("-vf", cmd)
        self.assertEqual(
            cmd[cmd.index("-vf") + 1],
            "scale=3840:2160:flags=lanczos:in_color_matrix=bt2020:out_color_matrix=bt2020",
        )

    @mock.patch("sdr2hdr.io.ffprobe_first_audio_codec", return_value=None)
    @mock.patch("sdr2hdr.io.subprocess.Popen")
    def test_open_encoder_builds_main10_qsv_command(
        self,
        popen_mock: mock.Mock,
        _: mock.Mock,
    ) -> None:
        info = VideoInfo(1920, 1080, 23.976, None, "yuv420p", 10.0, "progressive")
        open_encoder("output.mp4", "input.mp4", info, peak_nits=1000.0, encoder="hevc_qsv")
        cmd = popen_mock.call_args.args[0]
        codec_index = cmd.index("-c:v")
        self.assertEqual(cmd[codec_index + 1], "hevc_qsv")
        self.assertEqual(cmd[codec_index - 2 : codec_index], ["-pix_fmt", "p010le"])
        self.assertEqual(cmd[cmd.index("-profile:v") + 1], "main10")
        self.assertEqual(cmd[cmd.index("-preset") + 1], "slow")
        self.assertEqual(cmd[cmd.index("-global_quality") + 1], "18")
        self.assertNotIn("-tune", cmd)
        self.assertNotIn("-rc", cmd)

    @mock.patch("sdr2hdr.io.ffprobe_first_audio_codec", return_value=None)
    @mock.patch("sdr2hdr.io.subprocess.Popen")
    def test_open_encoder_keeps_libx265_unfiltered_without_resize(
        self,
        popen_mock: mock.Mock,
        _: mock.Mock,
    ) -> None:
        info = VideoInfo(1920, 1080, 23.976, None, "yuv420p", 10.0, "progressive")
        open_encoder("output.mp4", "input.mp4", info, peak_nits=1000.0, encoder="libx265")
        cmd = popen_mock.call_args.args[0]
        self.assertNotIn("-vf", cmd)


if __name__ == "__main__":
    unittest.main()
