import numpy as np
from stable_baselines3.common.logger import KVWriter

from human_feedback_rl.common.loggers import ExcludeFormatLogger, Logger, PrefixedLogger


class CapturingWriter(KVWriter):
    def __init__(self):
        self.writes = []

    def write(self, key_values, key_excluded, step=0):
        self.writes.append((dict(key_values), dict(key_excluded)))

    def close(self):
        pass


def _logger():
    writer = CapturingWriter()
    return Logger(folder=None, output_formats=[writer]), writer


def test_record_sum_accumulates():
    logger, writer = _logger()
    logger.record_sum("time/x", 1.0)
    logger.record_sum("time/x", 2.5)
    logger.dump()
    assert writer.writes[0][0]["time/x"] == 3.5


def test_dump_keys_flushes_only_requested_keys():
    logger, writer = _logger()
    logger.record("a", 1)
    logger.record("b", 2)
    logger.dump_keys(["a"])
    assert writer.writes[0][0] == {"a": 1}
    assert logger.name_to_value["b"] == 2  # untouched


def test_prefixed_logger_prefixes_and_flushes_own_keys_only():
    logger, writer = _logger()
    logger.record("other", 99)
    prefixed = PrefixedLogger(logger, "agent")
    prefixed.record("loss", 0.5)
    prefixed.dump()
    assert writer.writes[0][0] == {"agent/loss": 0.5}
    assert logger.name_to_value["other"] == 99


def test_exclude_format_logger_merges_excludes():
    logger, writer = _logger()
    excluded = ExcludeFormatLogger(logger, exclude="stdout")
    excluded.record("k", 1)
    excluded.record("j", 2, exclude="wandb")
    logger.dump()
    _, key_excluded = writer.writes[0]
    assert set(np.atleast_1d(key_excluded["k"])) == {"stdout"}
    assert set(key_excluded["j"]) == {"wandb", "stdout"}
