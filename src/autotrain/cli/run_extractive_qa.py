from argparse import ArgumentParser

from autotrain import logger
from autotrain.cli.utils import get_field_info
from autotrain.project import AutoTrainProject
from autotrain.trainers.extractive_question_answering.params import ExtractiveQuestionAnsweringParams

from . import BaseAutoTrainCommand


def run_extractive_qa_command_factory(args):
    return RunAutoTrainExtractiveQACommand(args)


class RunAutoTrainExtractiveQACommand(BaseAutoTrainCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = get_field_info(ExtractiveQuestionAnsweringParams)
        arg_list = [
            {
                "arg": "--train",
                "help": "Command to train the model",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--deploy",
                "help": "Command to deploy the model (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--inference",
                "help": "Command to run inference (limited availability)",
                "required": False,
                "action": "store_true",
            },
            {
                "arg": "--backend",
                "help": "Backend to use for training",
                "required": False,
                "default": "local",
            },
        ] + arg_list
        arg_list = [arg for arg in arg_list if arg["arg"] != "--disable-gradient-checkpointing"]
        run_extractive_qa_parser = parser.add_parser(
            "extractive-qa", description="✨ Run AutoTrain Extractive Question Answering"
        )
        for arg in arg_list:
            names = [arg["arg"]] + arg.get("alias", [])
            if "action" in arg:
                run_extractive_qa_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_extractive_qa_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                    choices=arg.get("choices"),
                )
        run_extractive_qa_parser.set_defaults(func=run_extractive_qa_command_factory)

    def __init__(self, args):
        self.args = args

        store_true_arg_names = [
            "train",
            "deploy",
            "inference",
            "auto_find_batch_size",
            "push_to_hub",
        ]
        for arg_name in store_true_arg_names:
            if getattr(self.args, arg_name) is None:
                setattr(self.args, arg_name, False)

        if self.args.train:
            if self.args.project_name is None:
                raise ValueError("Project name must be specified")
            if self.args.data_path is None:
                raise ValueError("Data path must be specified")
            if self.args.model is None:
                raise ValueError("Model must be specified")
            if self.args.push_to_hub:
                if self.args.username is None:
                    raise ValueError("Username must be specified for push to hub")
        else:
            raise ValueError("Must specify --train, --deploy or --inference")

    def run(self):
        logger.info("Running Extractive Question Answering")
        if self.args.train:
            params = ExtractiveQuestionAnsweringParams(**vars(self.args))
            project = AutoTrainProject(params=params, backend=self.args.backend, process=True)
            job_id = project.create()
            logger.info(f"Job ID: {job_id}")
