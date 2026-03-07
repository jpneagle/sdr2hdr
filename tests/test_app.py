import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from sdr2hdr.app import (
    CancelToken,
    ConversionCallbacks,
    ConversionRequest,
    build_output_path,
    default_encoder_for_platform,
    is_hardware_encoder_failure,
    is_videotoolbox_failure,
    run_conversion,
)
from sdr2hdr.io import has_expected_hdr_metadata


class AppTests(unittest.TestCase):
    def test_build_output_path_adds_hdr_suffix(self) -> None:
        self.assertEqual(build_output_path("/tmp/example.mp4"), "/tmp/example_hdr.mp4")

    def test_build_output_path_converts_transport_stream_extensions_to_mp4(self) -> None:
        self.assertEqual(build_output_path("/tmp/example.m2ts"), "/tmp/example_hdr.mp4")
        self.assertEqual(build_output_path("/tmp/example.ts"), "/tmp/example_hdr.mp4")

    def test_detects_videotoolbox_failure_message(self) -> None:
        self.assertTrue(is_videotoolbox_failure("Error: cannot create compression session: -12908"))
        self.assertTrue(is_videotoolbox_failure("hevc_videotoolbox failed"))
        self.assertFalse(is_videotoolbox_failure("generic libx265 failure"))

    def test_detects_hardware_encoder_failure_message(self) -> None:
        self.assertTrue(is_hardware_encoder_failure("hevc_videotoolbox failed"))
        self.assertTrue(is_hardware_encoder_failure("OpenEncodeSessionEx failed: unsupported device"))
        self.assertTrue(is_hardware_encoder_failure("Cannot load nvcuda.dll"))
        self.assertFalse(is_hardware_encoder_failure("generic libx265 failure"))

    def test_default_encoder_for_platform(self) -> None:
        self.assertEqual(default_encoder_for_platform("Darwin"), "hevc_videotoolbox")
        self.assertEqual(default_encoder_for_platform("Windows"), "hevc_nvenc")
        self.assertEqual(default_encoder_for_platform("Linux"), "libx265")

    def test_run_conversion_respects_cancel_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            token = CancelToken()
            token.cancel()
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
            )
            result = run_conversion(request, callbacks=ConversionCallbacks(), cancel_token=token)
            self.assertTrue(result.cancelled)
            self.assertEqual(result.processed_frames, 0)
            self.assertFalse(output_path.exists())

    def test_cancel_keeps_partial_output_after_some_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            token = CancelToken()
            events: list[int] = []

            def on_progress(processed: int, total: int | None, fps: float | None) -> None:
                events.append(processed)
                if processed >= 2:
                    token.cancel()

            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
            )
            result = run_conversion(
                request,
                callbacks=ConversionCallbacks(on_progress=on_progress),
                cancel_token=token,
            )
            self.assertTrue(result.cancelled)
            self.assertGreaterEqual(result.processed_frames, 2)
            self.assertTrue(output_path.exists())
            self.assertTrue(has_expected_hdr_metadata(str(output_path)))

    def test_cancel_can_drop_partial_output_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            token = CancelToken()
            token.cancel()
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
                keep_partial_output_on_cancel=False,
            )
            result = run_conversion(request, callbacks=ConversionCallbacks(), cancel_token=token)
            self.assertTrue(result.cancelled)
            self.assertFalse(output_path.exists())

    def test_completed_output_has_expected_hdr_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "input.mp4"
            output_path = Path(temp_dir) / "output.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=160x90:rate=24",
                    "-t",
                    "1",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    str(input_path),
                ],
                check=True,
            )
            request = ConversionRequest(
                input_path=str(input_path),
                output_path=str(output_path),
                preset="poc",
                encoder="libx265",
                backend="numpy",
                max_frames=12,
            )
            result = run_conversion(request, callbacks=ConversionCallbacks())
            self.assertFalse(result.cancelled)
            self.assertTrue(has_expected_hdr_metadata(str(output_path)))


if __name__ == "__main__":
    unittest.main()
