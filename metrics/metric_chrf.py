import evaluate

# In order to use it, add it to the SimulEval quality_scorer.py file.
# Requires evaluate and sacrebleu.

@register_quality_scorer("chrf")
class ChrfScorer(QualityScorer):
    """
    Compute chrF++ score

    Usage:
        :code:`--quality-metrics chrf`

    More info about evaluate chrf and default values:
    https://huggingface.co/spaces/evaluate-metric/chrf
    """
    def __init__(self, args) -> None:
        super().__init__()
        self.logger = logging.getLogger("simuleval.scorer.chrf")
        self.chrf = evaluate.load("chrf")

    def __call__(self, instances: Dict) -> float:
        try:
            return self.chrf.compute(
                predictions=[ins.prediction for ins in instances.values()],
                references=[[ins.reference] for ins in instances.values()],
                word_order=2,
            )["score"]
        except Exception as e:
            self.logger.error(str(e))
            return 0

    @classmethod
    def from_args(cls, args):
        return cls(args)
