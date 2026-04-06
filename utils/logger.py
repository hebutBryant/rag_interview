import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from utils.base import create_dir, get_date_now, print_text


class Logger:

    def __init__(self, log_path="./log.txt"):

        log_dir = os.path.dirname(log_path)
        if log_dir:
            create_dir(log_dir)

        self.info_dict = defaultdict(list)
        self.log_path = log_path

    # def update(self, info):
    #     self.info_dict.update(info)

    # def add(self, key, value):
    #     self.info_dict[key].append(value)

    # def get_times(self, key):
    #     return self.info_dict.get(key, None)

    # def all_info(self):
    #     return self.info_dict

    def log(self, *args: Any, oneline: bool = False, color="black") -> None:
        head = f"{get_date_now()} "
        msg = " ".join(map(str, args))

        # console output
        console_tail = "\r" if oneline else "\n"

        print(head, end="", flush=True)
        print_text(msg + console_tail, end="", color=color)

        # file output: always newline
        file_line = head + msg + "\n"
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(file_line)


if __name__ == "__main__":
    logger = Logger()
    logger.log("This is a test log.")
    logger.log("你好👋.")

    from utils.timer import Timer

    timer = Timer(skip=0)

    with timer.timing("update"):
        logger.log("Updating progress:", "25%", oneline=True, color="green")
        time.sleep(1)

    with timer.timing("update"):
        logger.log("Updating progress:", "75%", oneline=True, color="green")
        time.sleep(1)

    with timer.timing("update"):
        logger.log("Updating progress:", "100%", oneline=False, color="green")
        time.sleep(2)

    logger.log("Finished logging.")
    logger.log("\n")
    logger.log(timer.summary())

    # logger.add("hhhe", "123")

    # print(logger.all_info())
