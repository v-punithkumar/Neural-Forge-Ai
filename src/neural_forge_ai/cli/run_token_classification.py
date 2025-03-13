from argparse import ArgumentParser

from neural_forge_ai import logger
from neural_forge_ai.cli.utils import get_field_info
from neural_forge_ai.project import neural_forge_aiProject
from neural_forge_ai.trainers.token_classification.params import TokenClassificationParams

from . import Baseneural_forge_aiCommand


def run_token_classification_command_factory(args):
    return Runneural_forge_aiTokenClassificationCommand(args)


class Runneural_forge_aiTokenClassificationCommand(Baseneural_forge_aiCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        arg_list = get_field_info(TokenClassificationParams)
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
        arg_list = [arg for arg in arg_list if arg["arg"] != "--disable-gradient-checkpointing"]
        run_token_classification_parser = parser.add_parser(
            "token-classification", description="✨ Run neural_forge_ai Token Classification"
        )
        for arg in arg_list:
            names = [arg["arg"]] + arg.get("alias", [])
            if "action" in arg:
                run_token_classification_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    action=arg.get("action"),
                    default=arg.get("default"),
                )
            else:
                run_token_classification_parser.add_argument(
                    *names,
                    dest=arg["arg"].replace("--", "").replace("-", "_"),
                    help=arg["help"],
                    required=arg.get("required", False),
                    type=arg.get("type"),
                    default=arg.get("default"),
                    choices=arg.get("choices"),
                )
        run_token_classification_parser.set_defaults(func=run_token_classification_command_factory)

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
        logger.info("Running Token Classification")
        if self.args.train:
            params = TokenClassificationParams(**vars(self.args))
            project = neural_forge_aiProject(params=params, backend=self.args.backend, process=True)
            job_id = project.create()
            logger.info(f"Job ID: {job_id}")
