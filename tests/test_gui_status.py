import queue
import unittest
from types import SimpleNamespace

from sdr2hdr.gui import AppState, QueueJob, SDR2HDRGUI


class _FakeQueueView:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def get_children(self) -> list[str]:
        return [str(index) for index in range(len(self.rows))]

    def delete(self, *_: object) -> None:
        self.rows.clear()

    def insert(self, _parent: str, _where: str, iid: str, values: tuple[str, str, str]) -> None:
        self.rows.append(values)


class _FakeProgress:
    def __init__(self) -> None:
        self.config: dict[str, object] = {}
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True

    def configure(self, **kwargs: object) -> None:
        self.config.update(kwargs)


class _FakeStringVar:
    def __init__(self, value: str = "") -> None:
        self.value = value

    def set(self, value: str) -> None:
        self.value = value

    def get(self) -> str:
        return self.value


class _FakeRoot:
    def after(self, _delay: int, _callback: object) -> None:
        return


class GUIQueueStatusTests(unittest.TestCase):
    def test_refresh_job_list_uses_job_status_directly(self) -> None:
        app = SDR2HDRGUI.__new__(SDR2HDRGUI)
        app.queue_view = _FakeQueueView()
        app.queue_jobs = [
            QueueJob(request=SimpleNamespace(input_path="a.mp4", output_path="a_hdr.mp4"), status="starting")
        ]
        app.current_job_index = 0
        app.state = AppState.RUNNING

        SDR2HDRGUI._refresh_job_list(app)

        self.assertEqual(app.queue_view.rows[0][0], "STARTING")

    def test_progress_promotes_starting_job_to_running(self) -> None:
        app = SDR2HDRGUI.__new__(SDR2HDRGUI)
        app.root = _FakeRoot()
        app.queue_jobs = [
            QueueJob(request=SimpleNamespace(input_path="a.mp4", output_path="a_hdr.mp4"), status="starting")
        ]
        app.current_job_index = 0
        app.state = AppState.RUNNING
        app.event_queue = queue.Queue()
        app.event_queue.put(("progress", (12, 100, 23.5)))
        app.progress = _FakeProgress()
        app.progress_var = _FakeStringVar()
        app.status_var = _FakeStringVar()
        app._refresh_job_list = lambda: None
        app._set_job_status = lambda index, status: setattr(app.queue_jobs[index], "status", status)
        app._finish_current_job = lambda: None
        app._set_state = lambda state: setattr(app, "state", state)

        SDR2HDRGUI._drain_events(app)

        self.assertEqual(app.queue_jobs[0].status, "running")
        self.assertEqual(app.status_var.get(), "Converting")
        self.assertIn("value", app.progress.config)


if __name__ == "__main__":
    unittest.main()
