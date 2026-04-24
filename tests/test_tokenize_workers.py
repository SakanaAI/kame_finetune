import pytest

import tools.tokenize_audio as tokenize_audio
import tools.tokenize_oracle as tokenize_oracle


class DummyProcess:
    def __init__(self, exitcode):
        self.exitcode = exitcode


def test_tokenize_audio_raises_if_any_worker_failed():
    with pytest.raises(RuntimeError, match="worker 1 exited with code 1"):
        tokenize_audio._raise_if_any_worker_failed([DummyProcess(0), DummyProcess(1)])


def test_tokenize_oracle_raises_if_any_worker_failed():
    with pytest.raises(RuntimeError, match="worker 1 exited with code 1"):
        tokenize_oracle._raise_if_any_worker_failed([DummyProcess(0), DummyProcess(1)])
