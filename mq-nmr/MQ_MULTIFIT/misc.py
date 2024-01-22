class ProgressBar:
    def __init__(self, total = 100, prefix='‚', suffix='', decimals=1, length=50, print_end="\r"):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self.decimals = decimals
        self.length = length
        self.print_end = print_end

    def print_progress(self, iteration):
        percent = ("{0:." + str(self.decimals) + "f}").format(100 * (iteration / float(self.total)))
        filled_length = int(self.length * iteration // self.total)
        bar = "\033[1;38;5;87m" + "\u2588" * filled_length + "\033[0m" + '-' * (self.length - filled_length)
        print(f'\r   {self.prefix} |{bar}| {percent}% {self.suffix}', end=self.print_end)

        if iteration == self.total:
            print()

    @property
    def total(self):
        return self._total

    @total.setter
    def total(self, value):
        self._total = int(value)
    