from .base import OutputHandler
import mmcv


class SimpleHandler(OutputHandler):
    """Simple model outputs postprocess class."""
    def __init__(self, cfg):
        import os.path
        super(SimpleHandler, self).__init__()
        if cfg.get('work_dir', None) is None:
            cfg.work_dir = os.path.join('./work_dirs', 'outputs_dump')
        if cfg.get('results_dump_filename', None) is None:
            cfg.results_dump_filename = os.path.splitext(os.path.basename(cfg.filename))[0] + "_model_outputs.pkl"
        self.dump_file = os.path.join(cfg.work_dir, cfg.results_dump_filename)
        mmcv.mkdir_or_exist(cfg.work_dir)
        self.chunks_count = cfg.output_handler.get('chunks_count', None)

    def _process(self, outputs):
        """Simple postprocess method - dumps detector outputs to a file.
        Args:
            outputs (List[ndarray]):
                Standard model outputs
        Returns:
            outputs (List[ndarray])
                Standard model outputs
        """

        if self.chunks_count:
            max_chunk_length = len(outputs) // self.chunks_count
            for i in range(self.chunks_count + 1):
                mmcv.dump(outputs[max_chunk_length * i: max_chunk_length * (i+1)], ''.join(self.dump_file.split('.pkl')[:-1]) + 'chunk' + str(i) + '.pkl')
        else:
            mmcv.dump(outputs, self.dump_file)
        return outputs
