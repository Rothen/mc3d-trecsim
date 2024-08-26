from threading import Thread, Event


class Interval(Thread):
    def __init__(self, start_interval: float, interval: float, function, args=None, kwargs=None):
        Thread.__init__(self)
        self.start_interval = start_interval
        self.interval = interval
        self.function = function
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.finished = Event()

    def cancel(self):
        self.finished.set()

    def run(self):
        self.finished.wait(self.start_interval)
        self.function(*self.args, **self.kwargs)

        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

        self.finished.set()