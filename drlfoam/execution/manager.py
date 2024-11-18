"""
implements a class for handling the execution of runner for filling the buffer
"""
import logging

from queue import Queue
from threading import Thread

logger = logging.getLogger(__name__)


def string_args(args: list, kwargs: dict) -> str:
    args_str = ", ".join([str(arg) for arg in args])
    kwargs_str = ", ".join(f"{key}={str(value)}" for key, value in kwargs.items())
    if args_str and kwargs_str:
        return args_str + ", " + kwargs_str
    elif args_str and not kwargs_str:
        return args_str
    elif not args_str and kwargs_str:
        return kwargs_str
    else:
        return ""


class Runner(Thread):
    def __init__(self, tasks: Queue, name: str):
        super(Runner, self).__init__()
        self._tasks = tasks
        self._name = name
        self.daemon = True
        self.start()

    def run(self) -> None:
        while not self._tasks.empty():
            try:
                func, args, kwargs = self._tasks.get()
                logger.info(f"{self._name}: {func.__name__}({string_args(args, kwargs)})")
                func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"{self._name}: " + str(e))
            finally:
                self._tasks.task_done()

        logger.info(f"{self._name}: all tasks done")


class TaskManager(Queue):
    def __init__(self, n_runners_max: int):
        super(TaskManager, self).__init__()
        self._n_runners_max = n_runners_max
        self._runners = None

    def add(self, task, *args, **kwargs) -> None:
        self.put((task, args, kwargs))

    def run(self, wait: bool = True) -> None:
        n_runners = min(self._n_runners_max, self.qsize())
        self._runners = [Runner(self, f"Runner {i}") for i in range(n_runners)]
        if wait:
            self.wait()

    def wait(self) -> None:
        self.join()
