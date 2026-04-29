from evals.base import Evaluator


class KnowUnDoEvaluator(Evaluator):
    def __init__(self, eval_cfg, **kwargs):
        super().__init__("KnowUnDo", eval_cfg, **kwargs)
