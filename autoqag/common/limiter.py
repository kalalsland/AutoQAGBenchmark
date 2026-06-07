"""RPM / TPM 限流 (直接复用自 GraphGen graphgen/models/llm/limitter.py)。"""

import asyncio
import time
from datetime import datetime, timedelta

from autoqag.common.logging import logger


class RPM:
    def __init__(self, rpm: int = 1000):
        self.rpm = rpm
        self.record = {"rpm_slot": self.get_minute_slot(), "counter": 0}

    @staticmethod
    def get_minute_slot():
        current_time = time.time()
        dt_object = datetime.fromtimestamp(current_time)
        return dt_object.hour * 60 + dt_object.minute

    async def wait(self, silent=False):
        current = time.time()
        dt_object = datetime.fromtimestamp(current)
        minute_slot = self.get_minute_slot()

        if self.record["rpm_slot"] == minute_slot:
            if self.record["counter"] >= self.rpm:
                next_minute = dt_object.replace(second=0, microsecond=0) + timedelta(
                    minutes=1
                )
                sleep_time = abs(next_minute.timestamp() - current)
                if not silent:
                    logger.info("RPM sleep %s", sleep_time)
                await asyncio.sleep(sleep_time)
                self.record = {"rpm_slot": self.get_minute_slot(), "counter": 0}
        else:
            self.record = {"rpm_slot": self.get_minute_slot(), "counter": 0}
        self.record["counter"] += 1


class TPM:
    def __init__(self, tpm: int = 50000):
        self.tpm = tpm
        self.record = {"tpm_slot": self.get_minute_slot(), "counter": 0}

    @staticmethod
    def get_minute_slot():
        current_time = time.time()
        dt_object = datetime.fromtimestamp(current_time)
        return dt_object.hour * 60 + dt_object.minute

    async def wait(self, token_count, silent=False):
        current = time.time()
        dt_object = datetime.fromtimestamp(current)
        minute_slot = self.get_minute_slot()

        if self.record["tpm_slot"] != minute_slot:
            self.record = {"tpm_slot": minute_slot, "counter": token_count}
            return

        self.record["counter"] += token_count
        if self.record["counter"] > self.tpm:
            next_minute = dt_object.replace(second=0, microsecond=0) + timedelta(
                minutes=1
            )
            sleep_time = abs(next_minute.timestamp() - current)
            if not silent:
                logger.warning("TPM limit exceeded, wait %s seconds", sleep_time)
            await asyncio.sleep(sleep_time)
            self.record = {"tpm_slot": self.get_minute_slot(), "counter": token_count}
