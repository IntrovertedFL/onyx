import sys
import time
from collections.abc import Generator
from datetime import datetime

from onyx.connectors.interfaces import BaseConnector
from onyx.connectors.interfaces import CheckpointConnector
from onyx.connectors.interfaces import CheckpointOutput
from onyx.connectors.interfaces import LoadConnector
from onyx.connectors.interfaces import PollConnector
from onyx.connectors.models import ConnectorCheckpoint
from onyx.connectors.models import ConnectorFailure
from onyx.connectors.models import Document
from onyx.utils.logger import setup_logger


logger = setup_logger()


TimeRange = tuple[datetime, datetime]


def _wrap_checkpoint_output(
    checkpoint_connector_generator: CheckpointOutput,
) -> Generator[
    tuple[Document | None, ConnectorFailure | None, ConnectorCheckpoint | None],
    None,
    None,
]:
    next_checkpoint = None
    try:
        for document_or_failure in checkpoint_connector_generator:
            if isinstance(document_or_failure, Document):
                yield document_or_failure, None, None
            elif isinstance(document_or_failure, ConnectorFailure):
                yield None, document_or_failure, None
            else:
                raise ValueError(
                    f"Invalid document_or_failure type: {type(document_or_failure)}"
                )
    except StopIteration as e:
        next_checkpoint = e.value
        yield None, None, next_checkpoint

    if next_checkpoint is None:
        raise RuntimeError(
            "Checkpoint is None. This should never happen - the connector should always return a checkpoint."
        )


class ConnectorRunner:
    """
    Handles:
        - Batching
        - Additional exception logging
        - Combining different connector types to a single interface
    """

    def __init__(
        self,
        connector: BaseConnector,
        batch_size: int,
        time_range: TimeRange | None = None,
    ):
        self.connector = connector
        self.time_range = time_range
        self.batch_size = batch_size

        self.doc_batch: list[Document] = []

    def run(
        self, checkpoint: ConnectorCheckpoint
    ) -> Generator[
        tuple[
            list[Document] | None, ConnectorFailure | None, ConnectorCheckpoint | None
        ],
        None,
        None,
    ]:
        """Adds additional exception logging to the connector."""
        try:
            if isinstance(self.connector, CheckpointConnector):
                if self.time_range is None:
                    raise ValueError("time_range is required for CheckpointConnector")

                start = time.monotonic()
                checkpoint_connector_generator = self.connector.load_from_checkpoint(
                    start=self.time_range[0].timestamp(),
                    end=self.time_range[1].timestamp(),
                    checkpoint=checkpoint,
                )
                for document, failure, next_checkpoint in _wrap_checkpoint_output(
                    checkpoint_connector_generator
                ):
                    if document is not None:
                        self.doc_batch.append(document)

                    if failure is not None:
                        yield None, failure, None

                    if len(self.doc_batch) >= self.batch_size:
                        yield self.doc_batch, None, next_checkpoint
                        self.doc_batch = []

                # yield remaining documents
                if len(self.doc_batch) > 0:
                    yield self.doc_batch, None, next_checkpoint
                    self.doc_batch = []

                logger.debug(
                    f"Connector took {time.monotonic() - start} seconds to get to the next checkpoint."
                )

            else:
                finished_checkpoint = ConnectorCheckpoint.build_dummy_checkpoint()
                finished_checkpoint.has_more = False

                if isinstance(self.connector, PollConnector):
                    if self.time_range is None:
                        raise ValueError("time_range is required for PollConnector")

                    for document_batch in self.connector.poll_source(
                        start=self.time_range[0].timestamp(),
                        end=self.time_range[1].timestamp(),
                    ):
                        yield document_batch, None, None

                    yield None, None, finished_checkpoint
                elif isinstance(self.connector, LoadConnector):
                    for document_batch in self.connector.load_from_state():
                        yield document_batch, None, None

                    yield None, None, finished_checkpoint
                else:
                    raise ValueError(f"Invalid connector. type: {type(self.connector)}")
        except Exception:
            exc_type, _, exc_traceback = sys.exc_info()

            # Traverse the traceback to find the last frame where the exception was raised
            tb = exc_traceback
            if tb is None:
                logger.error("No traceback found for exception")
                raise

            while tb.tb_next:
                tb = tb.tb_next  # Move to the next frame in the traceback

            # Get the local variables from the frame where the exception occurred
            local_vars = tb.tb_frame.f_locals
            local_vars_str = "\n".join(
                f"{key}: {value}" for key, value in local_vars.items()
            )
            logger.error(
                f"Error in connector. type: {exc_type};\n"
                f"local_vars below -> \n{local_vars_str[:1024]}"
            )
            raise
