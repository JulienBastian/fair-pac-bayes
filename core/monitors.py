from tensorboardX import SummaryWriter

class MonitorMV:
    """
    A class for writing the relevant information in the console during the training and saving it afterward.
    """
    def __init__(self, logdir):
        super(MonitorMV, self).__init__()

        self.logdir = logdir
        self.writer = SummaryWriter(logdir)
        self.it = 0

    def write(self, it=None, **metrics):
        if it is None:
            it = self.it

        for key, item in metrics.items():
            self.writer.add_scalars(key, item, it)

        self.it += 1

    def close(self, logfile="monitor_scalars.json"):
        self.writer.export_scalars_to_json(self.logdir / logfile)
        self.writer.close()
