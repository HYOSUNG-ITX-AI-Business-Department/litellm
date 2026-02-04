"""
Security hook to prevent user B from seeing response from user A.

This hook uses the DBSpendUpdateWriter to batch-write response IDs to the database
instead of writing immediately on each request.
"""

from typing import TYPE_CHECKING, Any, AsyncGenerator, Dict, Optional, Tuple, Union, cast

from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import LitellmUserRoles
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.types.llms.openai import (
    BaseLiteLLMOpenAIResponseObject,
    ResponsesAPIResponse,
)
from litellm.types.utils import CallTypesLiteral, LLMResponseTypes, SpecialEnums

if TYPE_CHECKING:
    from litellm.caching.caching import DualCache
    from litellm.proxy._types import UserAPIKeyAuth


RESPONSE_ID_SECURITY_CACHE_PREFIX = "litellm:responses_id_security:response_id:"
RESPONSE_ID_SECURITY_CACHE_TTL_SECONDS = 60 * 60 * 24  # 24 hours


class ResponsesIDSecurity(CustomLogger):
    def __init__(self, internal_usage_cache: Optional[Any] = None):
        self.internal_usage_cache = internal_usage_cache

    async def async_pre_call_hook(
        self,
        user_api_key_dict: "UserAPIKeyAuth",
        cache: "DualCache",
        data: dict,
        call_type: CallTypesLiteral,
    ) -> Optional[Union[Exception, str, dict]]:
        # MAP all the responses api response ids to the encrypted response ids
        responses_api_call_types = {
            "aresponses",
            "aget_responses",
            "adelete_responses",
            "acancel_responses",
        }
        if call_type not in responses_api_call_types:
            return None
        if call_type == "aresponses":
            # check 'previous_response_id' if present in the data
            previous_response_id = data.get("previous_response_id")
            if previous_response_id:
                resolved_response_id = await self._resolve_encrypted_response_id(
                    response_id=previous_response_id,
                    user_api_key_dict=user_api_key_dict,
                )
                if resolved_response_id is not None:
                    data["previous_response_id"] = resolved_response_id
        elif call_type in {"aget_responses", "adelete_responses", "acancel_responses"}:
            response_id = data.get("response_id")
            if response_id:
                resolved_response_id = await self._resolve_encrypted_response_id(
                    response_id=response_id,
                    user_api_key_dict=user_api_key_dict,
                )
                if resolved_response_id is not None:
                    data["response_id"] = resolved_response_id
        return data

    async def _resolve_encrypted_response_id(
        self,
        response_id: str,
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> Optional[str]:
        if self._is_encrypted_response_id(response_id):
            original_response_id, user_id, team_id = self._decrypt_response_id(
                response_id
            )
            self.check_user_access_to_response_id(user_id, team_id, user_api_key_dict)
            return original_response_id

        cached_mapping = await self._get_cached_response_id_mapping(response_id)
        if cached_mapping is None:
            return None

        cached_user_id = cached_mapping.get("user_id")
        cached_team_id = cached_mapping.get("team_id")
        self.check_user_access_to_response_id(
            cached_user_id, cached_team_id, user_api_key_dict
        )
        return cast(Optional[str], cached_mapping.get("response_id"))

    def check_user_access_to_response_id(
        self,
        response_id_user_id: Optional[str],
        response_id_team_id: Optional[str],
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> bool:
        from litellm.proxy.proxy_server import general_settings

        if (
            user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value
            or user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN
        ):
            return True

        if response_id_user_id and response_id_user_id != user_api_key_dict.user_id:
            if general_settings.get("disable_responses_id_security", False):
                verbose_proxy_logger.debug(
                    f"Responses ID Security is disabled. User {user_api_key_dict.user_id} is accessing response id {response_id_user_id} which is not associated with them."
                )
                return True
            raise HTTPException(
                status_code=403,
                detail="Forbidden. The response id is not associated with the user, who this key belongs to. To disable this security feature, set general_settings::disable_responses_id_security to True in the config.yaml file.",
            )

        if response_id_team_id and response_id_team_id != user_api_key_dict.team_id:
            if general_settings.get("disable_responses_id_security", False):
                verbose_proxy_logger.debug(
                    f"Responses ID Security is disabled. Response belongs to team {response_id_team_id} but user {user_api_key_dict.user_id} is accessing it with team id {user_api_key_dict.team_id}."
                )
                return True
            raise HTTPException(
                status_code=403,
                detail="Forbidden. The response id is not associated with the team, who this key belongs to. To disable this security feature, set general_settings::disable_responses_id_security to True in the config.yaml file.",
            )

        return True

    def _is_encrypted_response_id(self, response_id: str) -> bool:
        split_result = response_id.split("resp_")
        if len(split_result) < 2:
            return False

        remaining_string = split_result[1]
        decrypted_value = decrypt_value_helper(
            value=remaining_string, key="response_id", return_original_value=True
        )

        if decrypted_value is None:
            return False

        if decrypted_value.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            return True
        return False

    def _decrypt_response_id(
        self, response_id: str
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Returns:
         - original_response_id: the original response id
         - user_id: the user id
         - team_id: the team id
        """
        split_result = response_id.split("resp_")
        if len(split_result) < 2:
            return response_id, None, None

        remaining_string = split_result[1]
        decrypted_value = decrypt_value_helper(
            value=remaining_string, key="response_id", return_original_value=True
        )

        if decrypted_value is None:
            return response_id, None, None

        if decrypted_value.startswith(SpecialEnums.LITELM_MANAGED_FILE_ID_PREFIX.value):
            # Expected format: "litellm_proxy:responses_api:response_id:{response_id};user_id:{user_id}"
            parts = decrypted_value.split(";")

            if len(parts) >= 2:
                # Extract response_id from "litellm_proxy:responses_api:response_id:{response_id}"
                response_id_part = parts[0]
                original_response_id = response_id_part.split("response_id:")[-1]

                # Extract user_id from "user_id:{user_id}"
                user_id_part = parts[1]
                user_id = user_id_part.split("user_id:")[-1]

                # Extract team_id from "team_id:{team_id}"
                team_id_part = parts[2]
                team_id = team_id_part.split("team_id:")[-1]

                return original_response_id, user_id, team_id
            else:
                # Fallback if format is unexpected
                return response_id, None, None
        return response_id, None, None

    def _get_signing_key(self) -> Optional[str]:
        """Get the signing key for encryption/decryption."""
        import os

        from litellm.proxy.proxy_server import master_key

        salt_key = os.getenv("LITELLM_SALT_KEY", None)
        if salt_key is None:
            salt_key = master_key
        return salt_key

    def _encrypt_response_id(
        self,
        response: BaseLiteLLMOpenAIResponseObject,
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> BaseLiteLLMOpenAIResponseObject:
        # encrypt the response id using the symmetric key
        # encrypt the response id, and encode the user id and response id in base64

        # Check if signing key is available
        signing_key = self._get_signing_key()
        if signing_key is None:
            verbose_proxy_logger.debug(
                "Response ID encryption is enabled but no signing key is configured. "
                "Please set LITELLM_SALT_KEY environment variable or configure a master_key. "
                "Skipping response ID encryption. "
                "See: https://docs.litellm.ai/docs/proxy/prod#5-set-litellm-salt-key"
            )
            return response

        response_id = getattr(response, "id", None)
        response_obj = getattr(response, "response", None)

        if (
            response_id
            and isinstance(response_id, str)
            and response_id.startswith("resp_")
        ):
            encrypted_response_id = SpecialEnums.LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR.value.format(
                response_id,
                user_api_key_dict.user_id or "",
                user_api_key_dict.team_id or "",
            )

            encoded_user_id_and_response_id = encrypt_value_helper(
                value=encrypted_response_id
            )
            setattr(
                response, "id", f"resp_{encoded_user_id_and_response_id}"
            )  # maintain the 'resp_' prefix for the responses api response id

        elif response_obj and isinstance(response_obj, ResponsesAPIResponse):
            encrypted_response_id = SpecialEnums.LITELLM_MANAGED_RESPONSE_API_RESPONSE_ID_COMPLETE_STR.value.format(
                response_obj.id,
                user_api_key_dict.user_id or "",
                user_api_key_dict.team_id or "",
            )
            encoded_user_id_and_response_id = encrypt_value_helper(
                value=encrypted_response_id
            )
            setattr(
                response_obj, "id", f"resp_{encoded_user_id_and_response_id}"
            )  # maintain the 'resp_' prefix for the responses api response id
            setattr(response, "response", response_obj)
        return response

    async def _cache_response_id_mapping(
        self,
        encrypted_response_id: str,
        original_response_id: str,
        user_api_key_dict: "UserAPIKeyAuth",
    ) -> None:
        if self.internal_usage_cache is None:
            return
        cache_key = f"{RESPONSE_ID_SECURITY_CACHE_PREFIX}{encrypted_response_id}"
        cache_value: Dict[str, Optional[str]] = {
            "response_id": original_response_id,
            "user_id": user_api_key_dict.user_id,
            "team_id": user_api_key_dict.team_id,
        }
        try:
            await self.internal_usage_cache.async_set_cache(
                key=cache_key,
                value=cache_value,
                ttl=RESPONSE_ID_SECURITY_CACHE_TTL_SECONDS,
                litellm_parent_otel_span=None,
            )
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Responses ID Security: unable to cache response id mapping - {str(e)}"
            )

    async def _get_cached_response_id_mapping(
        self, encrypted_response_id: str
    ) -> Optional[Dict[str, Any]]:
        if self.internal_usage_cache is None:
            return None
        cache_key = f"{RESPONSE_ID_SECURITY_CACHE_PREFIX}{encrypted_response_id}"
        try:
            return await self.internal_usage_cache.async_get_cache(
                key=cache_key,
                litellm_parent_otel_span=None,
            )
        except Exception as e:
            verbose_proxy_logger.debug(
                f"Responses ID Security: unable to fetch cached response id mapping - {str(e)}"
            )
            return None

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: "UserAPIKeyAuth",
        response: LLMResponseTypes,
    ) -> Any:
        """
        Queue response IDs for batch processing instead of writing directly to DB.

        This method adds response IDs to an in-memory queue, which are then
        batch-processed by the DBSpendUpdateWriter during regular database update cycles.
        """
        from litellm.proxy.proxy_server import general_settings

        if general_settings.get("disable_responses_id_security", False):
            return response
        if isinstance(response, ResponsesAPIResponse):
            original_response_id = self._get_primary_response_id(response=response)
            response = cast(
                ResponsesAPIResponse,
                self._encrypt_response_id(response, user_api_key_dict),
            )
            encrypted_response_id = self._get_primary_response_id(response=response)
            if (
                original_response_id
                and encrypted_response_id
                and original_response_id != encrypted_response_id
            ):
                await self._cache_response_id_mapping(
                    encrypted_response_id=encrypted_response_id,
                    original_response_id=original_response_id,
                    user_api_key_dict=user_api_key_dict,
                )
        return response

    async def async_post_call_streaming_iterator_hook(  # type: ignore
        self, user_api_key_dict: "UserAPIKeyAuth", response: Any, request_data: dict
    ) -> AsyncGenerator[BaseLiteLLMOpenAIResponseObject, None]:
        from litellm.proxy.proxy_server import general_settings

        async for chunk in response:
            if (
                isinstance(chunk, BaseLiteLLMOpenAIResponseObject)
                and user_api_key_dict.request_route
                == "/v1/responses"  # only encrypt the response id for the responses api
                and not general_settings.get("disable_responses_id_security", False)
            ):
                original_response_id = self._get_primary_response_id(response=chunk)
                chunk = self._encrypt_response_id(chunk, user_api_key_dict)
                encrypted_response_id = self._get_primary_response_id(response=chunk)
                if (
                    original_response_id
                    and encrypted_response_id
                    and original_response_id != encrypted_response_id
                ):
                    await self._cache_response_id_mapping(
                        encrypted_response_id=encrypted_response_id,
                        original_response_id=original_response_id,
                        user_api_key_dict=user_api_key_dict,
                    )
            yield chunk

    def _get_primary_response_id(
        self, response: BaseLiteLLMOpenAIResponseObject
    ) -> Optional[str]:
        response_id = getattr(response, "id", None)
        response_obj = getattr(response, "response", None)
        if (
            response_id
            and isinstance(response_id, str)
            and response_id.startswith("resp_")
        ):
            return response_id

        if response_obj and isinstance(response_obj, ResponsesAPIResponse):
            response_obj_id = getattr(response_obj, "id", None)
            if (
                response_obj_id
                and isinstance(response_obj_id, str)
                and response_obj_id.startswith("resp_")
            ):
                return response_obj_id
        return None
