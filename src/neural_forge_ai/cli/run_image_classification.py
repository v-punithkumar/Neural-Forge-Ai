from argparse import ArgumentParser

from neural_forge_ai import logger
from neural_forge_ai.cli.utils import get_field_info
from neural_forge_ai.project import neural_forge_aiProject
from neural_forge_ai.trainers.image_classification.params import ImageClassificationParams

from . import Baseneural_forge_aiCommand


def run_image_classification_command_factory(args):
    return Runneural_forge_aiImageClassificationCommand(args)


class Runneural_forge_aiImageClassificationCommand(Baseneural_forge_aiCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = get_field_info(ImageClassificationParams)
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
                "help": "Backend",
                "required": False,
                "type": str,
                "default": "local",
            },
        ] + arg_list
        run_image_classification_parser = parser.add_parser(
            "image-classification", description="✨ Run neural_forge_ai Image Classification"
        )
        for arg in arg_list:
            names = [arg["arg"]] + arg.get("alias", [])
            if "action" in arg:
                run_image_classification_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_image_classification_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                    choices=arg.get("choices"),
                )
        run_image_classification_parser.set_defaults(func=run_image_classification_command_factory)

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

        if self.args.backend.startswith("spaces") or self.args.backend.startswith("ep-"):
            if not self.args.push_to_hub:
                raise ValueError("Push to hub must be specified for spaces backend")
            if self.args.username is None:
                raise ValueError("Username must be specified for spaces backend")
            if self.args.token is None:
                raise ValueError("Token must be specified for spaces backend")

    def run(self):
        logger.info("Running Image Classification")
        if self.args.train:
            params = ImageClassificationParams(**vars(self.args))
            project = neural_forge_aiProject(params=params, backend=self.args.backend, process=True)
            job_id = project.create()
            logger.info(f"Job ID: {job_id}")
