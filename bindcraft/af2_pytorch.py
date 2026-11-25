from bindcraft.openfold.openfold.data import data_pipeline, templates
from bindcraft.prediction import DifferentiableProteinPredictor
from bindcraft.protein import ProteinStates, StructurePredictions
from openfold.openfold.model import AlphaFold
from openfold.openfold.config import model_config


class AlphaFoldPytorchDesignModel(DifferentiableProteinPredictor):
    def __init__(self):
        config = model_config('bindcraft_of2')
        self.model = AlphaFold(config)

        template_featurizer = templates.CustomHitFeaturizer(
            mmcif_dir=...,
            max_template_date="9999-12-31",
            max_hits=-1,
            kalign_binary_path=...,
        )

        monomer_data_pipeline = data_pipeline.DataPipeline(
            template_featurizer=template_featurizer,
        )

        data_processor = data_pipeline.DataPipelineMultimer(
            monomer_data_pipeline=monomer_data_pipeline
        )




    def predict(self, protein_states: ProteinStates, model: str | None=None) -> StructurePredictions:
        ...

    def sequence_gradients(self, protein_states: ProteinStates, losses: dict[str, DesignLoss], model: str | None=None, softmax_weight: float=1.0, one_hot_weight: float=0.0, temperature: float=1.0, logit_scale: float=2.0, reference_predictions: StructurePredictions | None=None) -> tuple[StructurePredictions, dict[str, Array], Array]:

