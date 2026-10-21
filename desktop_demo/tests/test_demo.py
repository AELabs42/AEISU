"""CPU-only checks of source selection, asset isolation and graceful lifecycle."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from demo.session import BACKEND, MODEL, ROOT, Session, backend_environment
from demo.windows import SourceWindow, StopEvent, source_windows


class DemoTests(unittest.TestCase):
    def test_model_identity(self):
        record = json.loads((ROOT / "clone_provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(), record["model_sha256"])
        self.assertFalse(record["frame_generation"])
        self.assertEqual([MODEL], list((ROOT / "native/models").rglob("*.bin")))

    def test_no_frame_generation_sources(self):
        self.assertFalse((ROOT / "native/src/framegen").exists())
        for path in (ROOT / "native/app").glob("*.inl"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("LSFG3", text)
            self.assertNotIn("FrameGenerationBackend", text)
        self.assertNotIn("--upscaler", (ROOT / "native/app/entry.inl").read_text(encoding="utf-8"))

    def test_sdr_only_shaders(self):
        shader = (ROOT / "native/src/d3d11_presenter.hlsl").read_text(encoding="utf-8")
        for removed in ("PqEotf", "PqOetf", "Rec2020ToRec709", "PsPresentHdr", "CsPresentDirectHdr"):
            self.assertNotIn(removed, shader)
        self.assertIn("float3 EncodeSourceToWorking(float3 color) { return saturate(color); }", shader)
        presenter = (ROOT / "native/app/presenter.inl").read_text(encoding="utf-8")
        self.assertIn("SwapChainFormatForMode() const { return kSdrSwapChainFormat; }", presenter)
        capture = (ROOT / "native/app/capture.inl").read_text(encoding="utf-8")
        self.assertNotIn("ApplyAutomaticHdrCalibration", capture)

    @unittest.skipUnless(BACKEND.is_file(), "Native backend has not been built")
    def test_native_sdr_policy(self):
        config = json.loads((BACKEND.parent / "runtime_paths.json").read_text(encoding="utf-8"))
        environment = os.environ.copy()
        environment["PATH"] = os.pathsep.join([config["cuda_bin"], config["cudnn_bin"], environment.get("PATH", "")])
        result = subprocess.run([str(BACKEND), "--self-test-sdr"], env=environment, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PQ and raw FP16 rejected", result.stdout)

    def test_source_label(self):
        self.assertEqual(SourceWindow(1, "Game", 1920, 1080).label, "Game  —  1920 × 1080")

    def test_portable_runtime_environment(self):
        with patch("pathlib.Path.read_text", return_value=json.dumps({"portable": True})):
            environment = backend_environment()
        self.assertEqual(environment["PATH"].split(os.pathsep)[0], str(BACKEND.parent))

    def test_default_capture_is_compositor_sdr(self):
        source = (ROOT / "native/app/common.inl").read_text(encoding="utf-8")
        self.assertIn("bool forceWgc = true;", source)
        self.assertIn("B8G8R8A8UIntNormalized", source)

    def test_nearest_toggle_routing(self):
        presenter = (ROOT / "native/app/presenter.inl").read_text(encoding="utf-8")
        session = (ROOT / "native/app/session.inl").read_text(encoding="utf-8")
        self.assertIn("UpscalerMode::BilinearComparison : UpscalerMode::NoUpscaler", presenter)
        self.assertIn("return RenderComparisonBypass(frame, geometry", presenter)
        self.assertIn("m_upscalerBypass && m_comparisonFilter == ComparisonFilter::Nearest", presenter)
        self.assertIn("kUpscalerBypassHotkeyId, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, VK_TAB", session)

    def test_enumeration(self):
        values = source_windows()
        self.assertTrue(all(window.width >= 64 and window.height >= 64 for window in values))
        self.assertTrue(all(isinstance(window.handle, int) for window in values))

    def test_stop_event(self):
        event = StopEvent("Local\\AEISUDemoUnitTestStop")
        event.signal()
        event.close()
        event.close()
        self.assertIsNone(event.handle)

    def test_stop_requests_graceful_shutdown(self):
        session = Session()
        session.process = MagicMock()
        session.process.poll.return_value = None
        session.stop_event = MagicMock()
        session.stop()
        session.stop_event.signal.assert_called_once()
        session.process.terminate.assert_not_called()

    def test_stop_hides_without_destroying_gpu_owner(self):
        text = (ROOT / "native/app/session.inl").read_text(encoding="utf-8")
        stop_function = text.split("void RequestOutputStop", 1)[1].split("void RequestUpscalerBypassToggle", 1)[0]
        self.assertIn("ShowWindow(hwnd, SW_HIDE)", stop_function)
        self.assertIn("PostMessageW(hwnd, kHideStoppedOutputMessage", stop_function)
        self.assertNotIn("DestroyWindow", stop_function)
        self.assertIn("IsWindowVisible(g_outputMouseHookWindow)", text)

    def test_handles_not_closed_while_running(self):
        session = Session()
        session.process = MagicMock()
        session.process.poll.return_value = None
        session.stop_event = MagicMock()
        session.close_handles()
        session.stop_event.close.assert_not_called()

    def test_invalid_window_rejected(self):
        session = Session()
        with patch("demo.session.BACKEND", MagicMock(is_file=MagicMock(return_value=True))), \
                patch("demo.session.USER32.IsWindow", return_value=False):
            with self.assertRaises(ValueError):
                session.start(0)

    def test_invalid_comparison_filter_rejected(self):
        with self.assertRaisesRegex(ValueError, "Comparison filter"):
            Session().start(0, comparison_filter="unsupported")

    def test_comparison_filter_launch_argument(self):
        for filter_name in ("nearest", "bilinear"):
            session = Session()
            with patch("demo.session.BACKEND", MagicMock(is_file=MagicMock(return_value=True))), \
                    patch("demo.session.backend_environment", return_value={}), \
                    patch("demo.session.USER32.IsWindow", return_value=True), \
                    patch("demo.session.StopEvent"), patch("demo.session.subprocess.Popen") as launch:
                launch.return_value.poll.return_value = 0
                session.start(1, comparison_filter=filter_name)
                command = launch.call_args.args[0]
                self.assertEqual(command[command.index("--comparison-filter") + 1], filter_name)
                session.close_handles()


if __name__ == "__main__":
    unittest.main()
