import time
from http import HTTPStatus

import httpx
from celery import shared_task
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded
from redis.lock import Lock as RedisLock
from tenacity import RetryError

from ee.onyx.server.tenants.product_gating import get_gated_tenants
from onyx.access.access import get_access_for_document
from onyx.background.celery.apps.app_base import task_logger
from onyx.background.celery.tasks.beat_schedule import BEAT_EXPIRES_DEFAULT
from onyx.background.celery.tasks.shared.RetryDocumentIndex import RetryDocumentIndex
from onyx.configs.constants import CELERY_GENERIC_BEAT_LOCK_TIMEOUT
from onyx.configs.constants import ONYX_CLOUD_TENANT_ID
from onyx.configs.constants import OnyxCeleryPriority
from onyx.configs.constants import OnyxCeleryTask
from onyx.configs.constants import OnyxRedisLocks
from onyx.db.connector_credential_pair import get_connector_credential_pair
from onyx.db.document import delete_document_by_connector_credential_pair__no_commit
from onyx.db.document import delete_documents_complete__no_commit
from onyx.db.document import fetch_chunk_count_for_document
from onyx.db.document import get_document
from onyx.db.document import get_document_connector_count
from onyx.db.document import mark_document_as_modified
from onyx.db.document import mark_document_as_synced
from onyx.db.document_external_access import batch_add_ext_perm_user_if_not_exists
from onyx.db.document_external_access import DocExternalAccess
from onyx.db.document_external_access import DocumentSource
from onyx.db.document_external_access import (
    upsert_document_by_connector_credential_pair,
)
from onyx.db.document_external_access import upsert_document_external_perms
from onyx.db.document_set import fetch_document_sets_for_document
from onyx.db.engine import get_all_tenant_ids
from onyx.db.engine import get_session_with_current_tenant
from onyx.db.search_settings import get_active_search_settings
from onyx.document_index.factory import get_default_document_index
from onyx.document_index.interfaces import VespaDocumentFields
from onyx.httpx.httpx_pool import HttpxPool
from onyx.redis.redis_connector_permission_sync import RedisConnectorPermissionSync
from onyx.redis.redis_connector_prune import RedisConnectorPrune
from onyx.redis.redis_pool import get_redis_client
from onyx.redis.redis_pool import redis_lock_dump
from onyx.server.documents.models import ConnectorCredentialPairIdentifier
from shared_configs.configs import IGNORED_SYNCING_TENANT_LIST

DOCUMENT_BY_CC_PAIR_CLEANUP_MAX_RETRIES = 3


# 5 seconds more than RetryDocumentIndex STOP_AFTER+MAX_WAIT
LIGHT_SOFT_TIME_LIMIT = 105
LIGHT_TIME_LIMIT = LIGHT_SOFT_TIME_LIMIT + 15

DOCUMENT_PERMISSIONS_UPDATE_MAX_RETRIES = 3


@shared_task(
    name=OnyxCeleryTask.DOCUMENT_BY_CC_PAIR_CLEANUP_TASK,
    soft_time_limit=LIGHT_SOFT_TIME_LIMIT,
    time_limit=LIGHT_TIME_LIMIT,
    max_retries=DOCUMENT_BY_CC_PAIR_CLEANUP_MAX_RETRIES,
    bind=True,
)
def document_by_cc_pair_cleanup_task(
    self: Task,
    document_id: str,
    connector_id: int,
    credential_id: int,
    tenant_id: str | None,
) -> bool:
    """A lightweight subtask used to clean up document to cc pair relationships.
    Created by connection deletion and connector pruning parent tasks."""

    """
    To delete a connector / credential pair:
    (1) find all documents associated with connector / credential pair where there
    this the is only connector / credential pair that has indexed it
    (2) delete all documents from document stores
    (3) delete all entries from postgres
    (4) find all documents associated with connector / credential pair where there
    are multiple connector / credential pairs that have indexed it
    (5) update document store entries to remove access associated with the
    connector / credential pair from the access list
    (6) delete all relevant entries from postgres
    """
    task_logger.debug(f"Task start: doc={document_id}")

    start = time.monotonic()

    try:
        with get_session_with_current_tenant() as db_session:
            cc_pair = get_connector_credential_pair(
                db_session=db_session,
                connector_id=connector_id,
                credential_id=credential_id,
            )
            if not cc_pair:
                task_logger.warning(
                    f"cc_pair not found for {connector_id} {credential_id}"
                )
                return False

            request_id = (
                self.request.id if self.request.id is not None else "missing-task-id"
            )

            RedisConnectorPrune.update_subtask_heartbeat(
                cc_pair.id, request_id, get_redis_client()
            )

            action = "skip"
            chunks_affected = 0

            active_search_settings = get_active_search_settings(db_session)
            doc_index = get_default_document_index(
                active_search_settings.primary,
                active_search_settings.secondary,
                httpx_client=HttpxPool.get("vespa"),
            )

            retry_index = RetryDocumentIndex(doc_index)

            count = get_document_connector_count(db_session, document_id)
            if count == 1:
                # count == 1 means this is the only remaining cc_pair reference to the doc
                # delete it from vespa and the db
                action = "delete"

                chunk_count = fetch_chunk_count_for_document(document_id, db_session)

                chunks_affected = retry_index.delete_single(
                    document_id,
                    tenant_id=tenant_id,
                    chunk_count=chunk_count,
                )

                delete_documents_complete__no_commit(
                    db_session=db_session,
                    document_ids=[document_id],
                )
            elif count > 1:
                action = "update"

                # count > 1 means the document still has cc_pair references
                doc = get_document(document_id, db_session)
                if not doc:
                    return False

                # the below functions do not include cc_pairs being deleted.
                # i.e. they will correctly omit access for the current cc_pair
                doc_access = get_access_for_document(
                    document_id=document_id, db_session=db_session
                )

                doc_sets = fetch_document_sets_for_document(document_id, db_session)
                update_doc_sets: set[str] = set(doc_sets)

                fields = VespaDocumentFields(
                    document_sets=update_doc_sets,
                    access=doc_access,
                    boost=doc.boost,
                    hidden=doc.hidden,
                )

                # update Vespa. OK if doc doesn't exist. Raises exception otherwise.
                chunks_affected = retry_index.update_single(
                    document_id,
                    tenant_id=tenant_id,
                    chunk_count=doc.chunk_count,
                    fields=fields,
                )

                # there are still other cc_pair references to the doc, so just resync to Vespa
                delete_document_by_connector_credential_pair__no_commit(
                    db_session=db_session,
                    document_id=document_id,
                    connector_credential_pair_identifier=ConnectorCredentialPairIdentifier(
                        connector_id=connector_id,
                        credential_id=credential_id,
                    ),
                )

                mark_document_as_synced(document_id, db_session)
            else:
                pass

            db_session.commit()

            elapsed = time.monotonic() - start
            task_logger.info(
                f"doc={document_id} "
                f"action={action} "
                f"refcount={count} "
                f"chunks={chunks_affected} "
                f"elapsed={elapsed:.2f}"
            )
    except SoftTimeLimitExceeded:
        task_logger.info(f"SoftTimeLimitExceeded exception. doc={document_id}")
        return False
    except Exception as ex:
        e: Exception | None = None
        if isinstance(ex, RetryError):
            task_logger.warning(
                f"Tenacity retry failed: num_attempts={ex.last_attempt.attempt_number}"
            )

            # only set the inner exception if it is of type Exception
            e_temp = ex.last_attempt.exception()
            if isinstance(e_temp, Exception):
                e = e_temp
        else:
            e = ex

        if isinstance(e, httpx.HTTPStatusError):
            if e.response.status_code == HTTPStatus.BAD_REQUEST:
                task_logger.exception(
                    f"Non-retryable HTTPStatusError: "
                    f"doc={document_id} "
                    f"status={e.response.status_code}"
                )
            return False

        task_logger.exception(f"Unexpected exception: doc={document_id}")

        if self.request.retries < DOCUMENT_BY_CC_PAIR_CLEANUP_MAX_RETRIES:
            # Still retrying. Exponential backoff from 2^4 to 2^6 ... i.e. 16, 32, 64
            countdown = 2 ** (self.request.retries + 4)
            self.retry(exc=e, countdown=countdown)
        else:
            # This is the last attempt! mark the document as dirty in the db so that it
            # eventually gets fixed out of band via stale document reconciliation
            task_logger.warning(
                f"Max celery task retries reached. Marking doc as dirty for reconciliation: "
                f"doc={document_id}"
            )
            with get_session_with_current_tenant() as db_session:
                # delete the cc pair relationship now and let reconciliation clean it up
                # in vespa
                delete_document_by_connector_credential_pair__no_commit(
                    db_session=db_session,
                    document_id=document_id,
                    connector_credential_pair_identifier=ConnectorCredentialPairIdentifier(
                        connector_id=connector_id,
                        credential_id=credential_id,
                    ),
                )
                mark_document_as_modified(document_id, db_session)
        return False

    # On completion, remove subtask ID from taskset
    request_id = self.request.id if self.request.id is not None else "missing-task-id"
    RedisConnectorPrune.remove_from_taskset(cc_pair.id, request_id, get_redis_client())
    return True


@shared_task(
    name=OnyxCeleryTask.CLOUD_BEAT_TASK_GENERATOR,
    ignore_result=True,
    trail=False,
    bind=True,
)
def cloud_beat_task_generator(
    self: Task,
    task_name: str,
    queue: str = OnyxCeleryTask.DEFAULT,
    priority: int = OnyxCeleryPriority.MEDIUM,
    expires: int = BEAT_EXPIRES_DEFAULT,
) -> bool | None:
    """a lightweight task used to kick off individual beat tasks per tenant."""
    time_start = time.monotonic()

    redis_client = get_redis_client(tenant_id=ONYX_CLOUD_TENANT_ID)

    lock_beat: RedisLock = redis_client.lock(
        f"{OnyxRedisLocks.CLOUD_BEAT_TASK_GENERATOR_LOCK}:{task_name}",
        timeout=CELERY_GENERIC_BEAT_LOCK_TIMEOUT,
    )

    # these tasks should never overlap
    if not lock_beat.acquire(blocking=False):
        return None

    last_lock_time = time.monotonic()
    tenant_ids: list[str] | list[None] = []

    try:
        tenant_ids = get_all_tenant_ids()
        gated_tenants = get_gated_tenants()
        for tenant_id in tenant_ids:
            if tenant_id in gated_tenants:
                continue

            current_time = time.monotonic()
            if current_time - last_lock_time >= (CELERY_GENERIC_BEAT_LOCK_TIMEOUT / 4):
                lock_beat.reacquire()
                last_lock_time = current_time

            # needed in the cloud
            if IGNORED_SYNCING_TENANT_LIST and tenant_id in IGNORED_SYNCING_TENANT_LIST:
                continue

            self.app.send_task(
                task_name,
                kwargs=dict(
                    tenant_id=tenant_id,
                ),
                queue=queue,
                priority=priority,
                expires=expires,
                ignore_result=True,
            )
    except SoftTimeLimitExceeded:
        task_logger.info(
            "Soft time limit exceeded, task is being terminated gracefully."
        )
    except Exception:
        task_logger.exception("Unexpected exception during cloud_beat_task_generator")
    finally:
        if not lock_beat.owned():
            task_logger.error(
                "cloud_beat_task_generator - Lock not owned on completion"
            )
            redis_lock_dump(lock_beat, redis_client)
        else:
            lock_beat.release()

    time_elapsed = time.monotonic() - time_start
    task_logger.info(
        f"cloud_beat_task_generator finished: "
        f"task={task_name} "
        f"num_tenants={len(tenant_ids)} "
        f"elapsed={time_elapsed:.2f}"
    )
    return True


@shared_task(
    name=OnyxCeleryTask.UPDATE_EXTERNAL_DOCUMENT_PERMISSIONS_TASK,
    soft_time_limit=LIGHT_SOFT_TIME_LIMIT,
    time_limit=LIGHT_TIME_LIMIT,
    max_retries=DOCUMENT_PERMISSIONS_UPDATE_MAX_RETRIES,
    bind=True,
)
def update_external_document_permissions_task(
    self: Task,
    tenant_id: str | None,
    serialized_doc_external_access: dict,
    source_string: str,
    connector_id: int,
    credential_id: int,
) -> bool:
    start = time.monotonic()

    document_external_access = DocExternalAccess.from_dict(
        serialized_doc_external_access
    )
    doc_id = document_external_access.doc_id
    external_access = document_external_access.external_access

    try:
        with get_session_with_current_tenant() as db_session:
            cc_pair = get_connector_credential_pair(
                db_session=db_session,
                connector_id=connector_id,
                credential_id=credential_id,
            )
            if not cc_pair:
                task_logger.warning(
                    f"cc_pair not found for {connector_id} {credential_id}"
                )
                return False

            # Update heartbeat upon task start
            RedisConnectorPermissionSync.update_subtask_heartbeat(
                cc_pair.id, self.request.id, get_redis_client()
            )

            # Add the users to the DB if they don't exist
            batch_add_ext_perm_user_if_not_exists(
                db_session=db_session,
                emails=list(external_access.external_user_emails),
                continue_on_error=True,
            )
            # Then upsert the document's external permissions
            created_new_doc = upsert_document_external_perms(
                db_session=db_session,
                doc_id=doc_id,
                external_access=external_access,
                source_type=DocumentSource(source_string),
            )

            if created_new_doc:
                # If a new document was created, we associate it with the cc_pair
                upsert_document_by_connector_credential_pair(
                    db_session=db_session,
                    connector_id=connector_id,
                    credential_id=credential_id,
                    document_ids=[doc_id],
                )

            elapsed = time.monotonic() - start
            task_logger.info(
                f"connector_id={connector_id} "
                f"doc={doc_id} "
                f"action=update_permissions "
                f"elapsed={elapsed:.2f}"
            )

    except Exception:
        task_logger.exception(
            f"Exception in update_external_document_permissions_task: "
            f"connector_id={connector_id} doc_id={doc_id}"
        )
        return False

    # On completion, remove subtask ID from taskset and creation hash
    RedisConnectorPermissionSync.remove_from_taskset(
        cc_pair.id, self.request.id, get_redis_client()
    )
    return True
