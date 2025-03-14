import argparse

from neural_forge_ai import __version__, logger
from neural_forge_ai.cli.run_api import Runneural_forge_aiAPICommand
from neural_forge_ai.cli.run_app import Runneural_forge_aiAppCommand
from neural_forge_ai.cli.run_extractive_qa import Runneural_forge_aiExtractiveQACommand
from neural_forge_ai.cli.run_image_classification import Runneural_forge_aiImageClassificationCommand
from neural_forge_ai.cli.run_image_regression import Runneural_forge_aiImageRegressionCommand
from neural_forge_ai.cli.run_llm import Runneural_forge_aiLLMCommand
from neural_forge_ai.cli.run_object_detection import Runneural_forge_aiObjectDetectionCommand
from neural_forge_ai.cli.run_sent_tranformers import Runneural_forge_aiSentenceTransformersCommand
from neural_forge_ai.cli.run_seq2seq import Runneural_forge_aiSeq2SeqCommand
from neural_forge_ai.cli.run_setup import RunSetupCommand
from neural_forge_ai.cli.run_spacerunner import Runneural_forge_aiSpaceRunnerCommand
from neural_forge_ai.cli.run_tabular import Runneural_forge_aiTabularCommand
from neural_forge_ai.cli.run_text_classification import Runneural_forge_aiTextClassificationCommand
from neural_forge_ai.cli.run_text_regression import Runneural_forge_aiTextRegressionCommand
from neural_forge_ai.cli.run_token_classification import Runneural_forge_aiTokenClassificationCommand
from neural_forge_ai.cli.run_tools import Runneural_forge_aiToolsCommand
from neural_forge_ai.parser import neural_forge_aiConfigParser


def main():
    parser = argparse.ArgumentParser(
        "neural_forge_ai advanced CLI",
        usage="neural_forge_ai <command> [<args>]",
        epilog="For more information about a command, run: `neural_forge_ai <command> --help`",
    )
    parser.add_argument("--version", "-v", help="Display neural_forge_ai version", action="store_true")
    parser.add_argument("--config", help="Optional configuration file", type=str)
    commands_parser = parser.add_subparsers(help="commands")

    # Register commands
    Runneural_forge_aiAppCommand.register_subcommand(commands_parser)
    Runneural_forge_aiLLMCommand.register_subcommand(commands_parser)
    RunSetupCommand.register_subcommand(commands_parser)
    Runneural_forge_aiAPICommand.register_subcommand(commands_parser)
    Runneural_forge_aiTextClassificationCommand.register_subcommand(commands_parser)
    Runneural_forge_aiImageClassificationCommand.register_subcommand(commands_parser)
    Runneural_forge_aiTabularCommand.register_subcommand(commands_parser)
    Runneural_forge_aiSpaceRunnerCommand.register_subcommand(commands_parser)
    Runneural_forge_aiSeq2SeqCommand.register_subcommand(commands_parser)
    Runneural_forge_aiTokenClassificationCommand.register_subcommand(commands_parser)
    Runneural_forge_aiToolsCommand.register_subcommand(commands_parser)
    Runneural_forge_aiTextRegressionCommand.register_subcommand(commands_parser)
    Runneural_forge_aiObjectDetectionCommand.register_subcommand(commands_parser)
    Runneural_forge_aiSentenceTransformersCommand.register_subcommand(commands_parser)
    Runneural_forge_aiImageRegressionCommand.register_subcommand(commands_parser)
    Runneural_forge_aiExtractiveQACommand.register_subcommand(commands_parser)

    args = parser.parse_args()

    if args.version:
        print(__version__)
        exit(0)

    if args.config:
        logger.info(f"Using neural_forge_ai configuration: {args.config}")
        cp = neural_forge_aiConfigParser(args.config)
        cp.run()
        exit(0)

    if not hasattr(args, "func"):
        parser.print_help()
        exit(1)

    command = args.func(args)
    command.run()


if __name__ == "__main__":
    main()
