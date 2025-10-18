import os
import sagemaker
import boto3
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat,
)
from sagemaker.tensorflow import TensorFlow
from sagemaker.tensorflow.processing import TensorFlowProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, TransformStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.transformer import Transformer
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.condition_step import ConditionStep, JsonGet
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join
from sagemaker.inputs import TransformInput
from sagemaker import image_uris, Session

# --------------------------------------------------------------------------------------------------------------------
# Session/role/region
# --------------------------------------------------------------------------------------------------------------------
sagemaker_session = sagemaker.session.Session()
region = sagemaker_session.boto_region_name
role = sagemaker.get_execution_role()
pipeline_session = PipelineSession()
default_bucket = sagemaker_session.default_bucket()

# --------------------------------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------------------------------
S3_JSON_URI = ParameterString(
    name="S3JsonUri",
    default_value="s3://doom-cnn-frames",
)

S3_IMAGES_URI = ParameterString(
    name="S3ImagesUri",
    default_value="s3://doom-cnn-frames",
)

IMAGE_HEIGHT = ParameterInteger("ImageHeight", default_value=160)
IMAGE_WIDTH  = ParameterInteger("ImageWidth", default_value=120)
JPEG_QUALITY = ParameterInteger("JpegQuality", default_value=90)

TRAIN_INSTANCE_TYPE = ParameterString("TrainInstanceType", default_value="ml.m5.xlarge")
TRAIN_EPOCHS        = ParameterInteger("TrainEpochs", default_value=20)
TRAIN_BATCH_SIZE    = ParameterInteger("TrainBatchSize", default_value=32)
LEARNING_RATE       = ParameterFloat("LearningRate", default_value=1e-3)

PROCESS_INSTANCE_TYPE = ParameterString("ProcessInstanceType", default_value="ml.m5.xlarge")
TRANSFORM_INSTANCE_TYPE = ParameterString("TransformInstanceType", default_value="ml.m5.xlarge")
MODEL_APPROVAL_STATUS = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")

MODEL_PACKAGE_GROUP = ParameterString(
    "ModelPackageGroupName",
    default_value="DoomActionPredictor",
)

F1_THRESHOLD = ParameterFloat("F1Threshold", default_value=0.85)

# --------------------------------------------------------------------------------------------------------------------
# Processing Step (load JSONs and IMAGES from separate S3 prefixes)
# --------------------------------------------------------------------------------------------------------------------
tf_processor = TensorFlowProcessor(
    framework_version="2.13",
    py_version="py310",
    role=role,
    instance_type=PROCESS_INSTANCE_TYPE,
    instance_count=1,
    base_job_name="doom-preprocess",
    sagemaker_session=pipeline_session,
)

preprocess_step_args = tf_processor.run(
    code="src/preprocess.py",
    inputs=[
        ProcessingInput(
            source=S3_JSON_URI,
            destination="/opt/ml/processing/json",
            input_name="json",
        ),
        ProcessingInput(
            source=S3_IMAGES_URI,
            destination="/opt/ml/processing/images",
            input_name="images",
        ),
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
        ProcessingOutput(output_name="val",   source="/opt/ml/processing/output/val"),
        ProcessingOutput(output_name="meta",  source="/opt/ml/processing/output/meta"),
    ],
    arguments=[
        "--img_height",  IMAGE_HEIGHT.to_string(),
        "--img_width",   IMAGE_WIDTH.to_string(),
        "--jpeg_quality", JPEG_QUALITY.to_string(),
    ],
)

processing_step = ProcessingStep(
    name="DoomPreprocess",
    step_args=preprocess_step_args,
)

# --------------------------------------------------------------------------------------------------------------------
# Training Step (TensorFlow Keras)
# --------------------------------------------------------------------------------------------------------------------
img = image_uris.retrieve("tensorflow", region=region, version="2.13", py_version="py310", image_scope="training", instance_type="ml.m5.xlarge")

tf_estimator = TensorFlow(
    entry_point="src/train.py",
    role=role,
    instance_type=TRAIN_INSTANCE_TYPE,
    instance_count=1,
    framework_version="2.13",
    py_version="py310",
    sagemaker_session=pipeline_session,
    image_uri=img,
    hyperparameters={
        "epochs": TRAIN_EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "img_height": IMAGE_HEIGHT,
        "img_width": IMAGE_WIDTH,
        "num_actions": 3,
    },
    base_job_name="doom-train",
)

step_train = TrainingStep(
    name="DoomTrain",
    estimator=tf_estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "val": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri,
            content_type="text/csv",
        ),
        "meta": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["meta"].S3Output.S3Uri,
            content_type="application/json",
        ),
    },
)

# --------------------------------------------------------------------------------------------------------------------
# Evaluation Step (reads model + val set => evaluation.json)
# --------------------------------------------------------------------------------------------------------------------
eval_processor = TensorFlowProcessor(
    framework_version="2.13",
    py_version="py310",
    role=role,
    instance_type=PROCESS_INSTANCE_TYPE,
    instance_count=1,
    base_job_name="doom-evaluate",
    sagemaker_session=pipeline_session,
)

evaluation_report = PropertyFile(
    name="DoomEvaluationReport",
    output_name="evaluation",
    path="evaluation.json",
)

eval_step_args = eval_processor.run(
    code="src/evaluate.py",
    inputs=[
        ProcessingInput(
            source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model",
            input_name="model",
        ),
        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["val"].S3Output.S3Uri,
            destination="/opt/ml/processing/val",
            input_name="val",
        ),
        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["meta"].S3Output.S3Uri,
            destination="/opt/ml/processing/meta",
            input_name="meta",
        ),
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ],
)

step_eval = ProcessingStep(
    name="DoomEvaluate",
    step_args=eval_step_args,
    property_files=[evaluation_report],
)

# Build Model Metrics from the evaluation output
eval_json_s3 = Join(
    on="/",
    values=[
        step_eval.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
        "evaluation.json",
    ],
)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=eval_json_s3,
        content_type="application/json",
    )
)

# --------------------------------------------------------------------------------------------------------------------
# Register Model to Model Registry
# --------------------------------------------------------------------------------------------------------------------
register_step = RegisterModel(
    name="DoomRegisterModel",
    estimator=tf_estimator,
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["application/json"],
    inference_instances=["ml.g4dn.xlarge", "ml.g6.xlarge", "ml.m5.large"],
    transform_instances=["ml.m5.large"],
    model_package_group_name=MODEL_PACKAGE_GROUP,
    model_metrics=model_metrics,
    approval_status=MODEL_APPROVAL_STATUS,
    image_uri=tf_estimator.training_image_uri(),
)

# --------------------------------------------------------------------------------------------------------------------
# Create Model + (optional) Batch Transform (post-registration sanity check)
# --------------------------------------------------------------------------------------------------------------------
model = Model(
    image_uri=tf_estimator.training_image_uri(),
    model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=pipeline_session,
)

step_create_model = ModelStep(
    name="DoomCreateModel",
    step_args=model.create(instance_type="ml.m5.large"),
)

transformer = Transformer(
    model_name=step_create_model.properties.ModelName,
    instance_type=TRANSFORM_INSTANCE_TYPE,
    instance_count=1,
    output_path=f"s3://{default_bucket}/doom-transform-output",
)

# Dummy transform on the validation CSV (structure-compatible)
step_transform = TransformStep(
    name="DoomBatchTransform",
    transformer=transformer,
    inputs=TransformInput(
        data=processing_step.properties.ProcessingOutputConfig
            .Outputs["val"].S3Output.S3Uri,
        content_type="text/csv",   # optional but sensible
        split_type="Line",         # batch transform reads line-by-line
    ),
)


# --------------------------------------------------------------------------------------------------------------------
# Hard gate on macro F1 from evaluation.json (fail pipeline if below threshold)
# --------------------------------------------------------------------------------------------------------------------
macro_f1 = JsonGet(
    step_eval,
    evaluation_report,
    "binary_classification_metrics.macro_avg.f1",
)

fail_step = FailStep(
    name="MetricsBelowThreshold",
    error_message="Macro F1 is below the required threshold"
)

quality_gate = ConditionStep(
    name="QualityGate",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=macro_f1,
            right=F1_THRESHOLD
        )
    ],
    if_steps=[register_step, step_create_model, step_transform],
    else_steps=[fail_step],
)

# --------------------------------------------------------------------------------------------------------------------
# Assemble pipeline
# --------------------------------------------------------------------------------------------------------------------
pipeline = Pipeline(
    name="DoomTrainingPipeline",
    parameters=[
        S3_JSON_URI, S3_IMAGES_URI,
        IMAGE_HEIGHT, IMAGE_WIDTH, JPEG_QUALITY,
        TRAIN_INSTANCE_TYPE, TRAIN_EPOCHS, TRAIN_BATCH_SIZE, LEARNING_RATE,
        PROCESS_INSTANCE_TYPE, TRANSFORM_INSTANCE_TYPE, MODEL_APPROVAL_STATUS,
        MODEL_PACKAGE_GROUP, F1_THRESHOLD
    ],
    steps=[processing_step, step_train, step_eval, quality_gate],
    sagemaker_session=pipeline_session,
)

def main():
    _ = pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    print("Execution ARN:", execution.arn)
    try:
        execution.wait()
    except Exception as e:
        print("Pipeline execution error:", e)
    print("Steps:", execution.list_steps())

if __name__ == "__main__":
    main()